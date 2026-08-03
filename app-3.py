from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
import zipfile
from itertools import combinations
from typing import Any, BinaryIO, Iterable, TextIO
import re
import unicodedata

import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---- config.py ----
APP_NAME = 'BFTD Keeper Lab'
APP_VERSION = '2.4.0'
DEFAULT_LEAGUE_ID = '1339101169082990592'
SKILL_POSITIONS = ('QB', 'RB', 'WR', 'TE', 'K', 'DEF')
CORE_POSITIONS = ('QB', 'RB', 'WR', 'TE')

@dataclass(frozen=True)
class KeeperRules:
    keeper_count: int = 5
    max_qb: int = 1
    min_second_year: int = 1
KEEPER_RULES = KeeperRules()

# ---- name_matching.py ----
_SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}
_COMMON_ALIASES = {'gabe davis': 'gabriel davis', 'ken walker': 'kenneth walker', 'mike pittman': 'michael pittman', 'hollywood brown': 'marquise brown', 'm brown': 'marquise brown', 'tank dell': 'nathaniel dell', 't dell': 'nathaniel dell', 'r pearsall': 'ricky pearsall', 'chig okonkwo': 'chigoziem okonkwo', 'dj moore': 'd j moore', 'dk metcalf': 'd k metcalf', 'tj hockenson': 't j hockenson'}

def normalize_name(value: object) -> str:
    if value is None:
        return ''
    text = unicodedata.normalize('NFKD', str(value))
    text = ''.join((char for char in text if not unicodedata.combining(char)))
    text = text.lower().replace('’', "'")
    text = re.sub('[^a-z0-9 ]+', ' ', text)
    tokens = [token for token in text.split() if token not in _SUFFIXES]
    normalized = ' '.join(tokens)
    return _COMMON_ALIASES.get(normalized, normalized)


_TEAM_ALIASES = {
    'JAC': 'JAX', 'GBP': 'GB', 'KCC': 'KC', 'LVR': 'LV', 'NOS': 'NO',
    'NEP': 'NE', 'SFO': 'SF', 'TBB': 'TB', 'WSH': 'WAS', 'LA': 'LAR',
}

def normalize_position(value: object) -> str:
    if value is None or pd.isna(value):
        return ''
    text = str(value).upper().strip().replace('D/ST', 'DST')
    match = re.match(r'^(QB|RB|WR|TE|K|DST|DEF)', text)
    if not match:
        return text
    position = match.group(1)
    return 'DEF' if position == 'DST' else position

def normalize_team(value: object) -> str:
    if value is None or pd.isna(value):
        return ''
    text = str(value).upper().strip()
    if text in {'', 'NAN', '<NA>', 'NONE'}:
        return ''
    return _TEAM_ALIASES.get(text, text)

def _clean_external_player_name(value: object) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return '', ''
    text = str(value).strip()
    # FantasyPros commonly places the NFL team at the end of the player field,
    # e.g. "J. Allen (BUF)". Strip it for matching and preserve it separately.
    team_match = re.search(r'\(([A-Z]{2,3}|FA)\)\s*$', text, flags=re.I)
    team = normalize_team(team_match.group(1)) if team_match else ''
    if team_match:
        text = text[:team_match.start()].strip()
    text = re.sub(r'\s+(Q|O|IR|PUP|SUSP|D|NA)\s*$', '', text, flags=re.I).strip()
    return text, team

def _initial_last_key(value: object) -> str:
    tokens = normalize_name(value).split()
    if len(tokens) < 2:
        return ''
    return f'{tokens[0][0]} {tokens[-1]}'

# ---- keeper_optimizer.py ----
@dataclass
class KeeperOption:
    keepers: pd.DataFrame
    total_score: float
    qb_count: int
    second_year_count: int

@dataclass
class KeeperResult:
    feasible: bool
    keepers: pd.DataFrame
    total_score: float
    qb_count: int
    second_year_count: int
    explanation: str
    alternatives: list[KeeperOption] = field(default_factory=list)
    unrestricted_score: float = 0.0
    rule_cost: float = 0.0

def _bool_series(data: pd.DataFrame, column: str, default: bool=False) -> pd.Series:
    if column not in data.columns:
        return pd.Series(default, index=data.index, dtype=bool)
    values = data[column]
    if values.dtype == bool:
        return values.fillna(default)
    return values.astype(str).str.strip().str.lower().isin({'true', '1', 'yes', 'y', 'x'})

def _second_year_series(data: pd.DataFrame) -> pd.Series:
    if 'is_second_year' in data.columns:
        return _bool_series(data, 'is_second_year')
    years = pd.to_numeric(data.get('years_exp'), errors='coerce')
    return years.eq(1)

def optimize_keepers(team_df: pd.DataFrame, rules: KeeperRules, *, top_n: int=5) -> KeeperResult:
    data = team_df.copy().reset_index(drop=True)
    data['optimizer_score'] = pd.to_numeric(data.get('optimizer_score'), errors='coerce')
    data['years_exp'] = pd.to_numeric(data.get('years_exp'), errors='coerce')
    data['position'] = data.get('position', pd.Series(index=data.index, dtype='string')).astype('string').str.upper()
    data['is_second_year'] = _second_year_series(data)
    data['locked'] = _bool_series(data, 'locked')
    data['excluded'] = _bool_series(data, 'excluded')
    if bool((data['locked'] & data['excluded']).any()):
        names = data.loc[data['locked'] & data['excluded'], 'player_name'].astype(str).tolist()
        return KeeperResult(False, pd.DataFrame(), 0.0, 0, 0, f"A player cannot be both locked and excluded: {', '.join(names)}.")
    eligible = data[data['optimizer_score'].notna() & ~data['excluded']].copy().reset_index(drop=True)
    locked = eligible[eligible['locked']].copy()
    if len(locked) > rules.keeper_count:
        return KeeperResult(False, pd.DataFrame(), 0.0, 0, 0, f'{len(locked)} players are locked, but only {rules.keeper_count} keepers are allowed.')
    if int((locked['position'] == 'QB').sum()) > rules.max_qb:
        return KeeperResult(False, pd.DataFrame(), 0.0, 0, 0, f'The locked group contains more than {rules.max_qb} quarterback.')
    if len(eligible) < rules.keeper_count:
        return KeeperResult(False, pd.DataFrame(), 0.0, 0, 0, f'Only {len(eligible)} non-excluded rostered players have usable ranking/value data. At least {rules.keeper_count} are required.')
    locked_indices = tuple(eligible.index[eligible['locked']].tolist())
    available_indices = [idx for idx in eligible.index if idx not in locked_indices]
    choose_count = rules.keeper_count - len(locked_indices)
    unrestricted_candidates = eligible.sort_values('optimizer_score', ascending=False)
    unrestricted = pd.concat([locked, unrestricted_candidates[~unrestricted_candidates.index.isin(locked_indices)]], axis=0).drop_duplicates(subset=['player_id'] if 'player_id' in eligible.columns else None).head(rules.keeper_count)
    unrestricted_score = float(unrestricted['optimizer_score'].sum())
    options: list[KeeperOption] = []
    for chosen in combinations(available_indices, choose_count):
        combo = tuple(locked_indices) + tuple(chosen)
        subset = eligible.loc[list(combo)]
        qb_count = int((subset['position'] == 'QB').sum())
        second_year_count = int(subset['is_second_year'].sum())
        if qb_count > rules.max_qb or second_year_count < rules.min_second_year:
            continue
        score = float(subset['optimizer_score'].sum())
        options.append(KeeperOption(keepers=subset.sort_values('optimizer_score', ascending=False).reset_index(drop=True), total_score=score, qb_count=qb_count, second_year_count=second_year_count))
    if not options:
        second_year_names = eligible.loc[eligible['is_second_year'], 'player_name'].astype(str).tolist()
        return KeeperResult(False, pd.DataFrame(), 0.0, 0, 0, f"No legal five-player combination exists with the current rankings, locks, exclusions, and eligibility flags. Detected second-year candidates: {', '.join(second_year_names) or 'none'}.", unrestricted_score=unrestricted_score)
    options.sort(key=lambda option: option.total_score, reverse=True)
    best = options[0]
    rule_cost = max(0.0, unrestricted_score - best.total_score)
    explanation = 'Highest-scoring legal combination using the uploaded ranking/value exactly as supplied. No quarterback, Superflex, positional-scarcity, or roster-need multiplier was added.'
    return KeeperResult(feasible=True, keepers=best.keepers, total_score=best.total_score, qb_count=best.qb_count, second_year_count=best.second_year_count, explanation=explanation, alternatives=options[1:max(1, top_n)], unrestricted_score=unrestricted_score, rule_cost=rule_cost)

# ---- external_rankings.py ----
@dataclass(frozen=True)
class MatchReport:
    total_rostered: int
    matched: int
    unmatched_names: tuple[str, ...] = ()
    method_counts: tuple[tuple[str, int], ...] = ()
    source_rows: int = 0
COLUMN_ALIASES = {'player_name': ['player_name', 'player', 'name', 'player name', 'full_name', 'full name'], 'external_rank': ['rank', 'rk', 'overall rank', 'overall_rank', 'ecr', 'overall', 'redraft rank', 'superflex rank', 'sf rank'], 'external_value': ['value', 'trade value', 'player value', 'score', 'points', 'fantasycalc value', 'superflex value', 'sf value'], 'position': ['position', 'pos', 'player position'], 'nfl_team': ['team', 'nfl team', 'nfl_team', 'tm'], 'external_age': ['age', 'player age'], 'external_years_exp': ['years_exp', 'years exp', 'experience', 'exp'], 'sleeper_id': ['sleeper_id', 'sleeper id', 'sleeperid'], 'source_adp': ['adp', 'average draft position'], 'tier': ['tier']}

def _uploaded_bytes(file_obj: BinaryIO | TextIO) -> tuple[bytes, str]:
    name = str(getattr(file_obj, 'name', '') or '')
    if hasattr(file_obj, 'getvalue'):
        payload = file_obj.getvalue()
    else:
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
        payload = file_obj.read()
    if isinstance(payload, str):
        payload = payload.encode('utf-8')
    return bytes(payload), name

def _read_csv_payload(payload: bytes) -> pd.DataFrame:
    if payload.lstrip().lower().startswith((b'<!doctype html', b'<html')):
        raise ValueError('This looks like a saved webpage, not a rankings file. Download the actual CSV or Excel file.')
    last_error: Exception | None = None
    for encoding in ('utf-8-sig', 'utf-8', 'latin-1'):
        try:
            frame = pd.read_csv(BytesIO(payload), encoding=encoding)
            if len(frame.columns) == 1:
                detected = pd.read_csv(BytesIO(payload), encoding=encoding, sep=None, engine='python')
                if len(detected.columns) > 1:
                    frame = detected
            return frame
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError(f'Could not read the CSV file: {last_error}')

def _read_uploaded_table(file_obj: BinaryIO | TextIO) -> pd.DataFrame:
    payload, original_name = _uploaded_bytes(file_obj)
    if not payload:
        raise ValueError('The selected file is empty.')
    suffix = Path(original_name).suffix.lower()

    # Android sometimes supplies a generic MIME type. We therefore inspect the
    # file contents instead of relying on the browser-reported type.
    is_zip_container = zipfile.is_zipfile(BytesIO(payload))
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(BytesIO(payload))
    if is_zip_container:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            names = [name for name in archive.namelist() if not name.endswith('/')]
            if any(name.startswith('xl/') for name in names):
                return pd.read_excel(BytesIO(payload))
            candidates = [name for name in names if Path(name).suffix.lower() in {'.csv', '.xlsx', '.xls'}]
            if not candidates:
                raise ValueError('The ZIP does not contain a CSV or Excel file.')
            candidates.sort(key=lambda name: (Path(name).suffix.lower() != '.csv', name.lower()))
            selected = candidates[0]
            inner = archive.read(selected)
            if Path(selected).suffix.lower() == '.csv':
                return _read_csv_payload(inner)
            return pd.read_excel(BytesIO(inner))
    return _read_csv_payload(payload)

def _rankings_read_csv(file_obj: BinaryIO | TextIO) -> pd.DataFrame:
    return _read_uploaded_table(file_obj)

def _rankings_norm_col(value: object) -> str:
    return ' '.join(str(value).strip().lower().replace('_', ' ').split())

def _rankings_find_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {_rankings_norm_col(col): col for col in columns}
    for alias in aliases:
        if _rankings_norm_col(alias) in normalized:
            return normalized[_rankings_norm_col(alias)]
    for normalized_name, original in normalized.items():
        if any((_rankings_norm_col(alias) in normalized_name for alias in aliases)):
            return original
    return None

def load_rankings_csv(file_obj: BinaryIO | TextIO) -> pd.DataFrame:
    raw = _rankings_read_csv(file_obj)
    if raw.empty:
        raise ValueError('The rankings file is empty.')
    rename: dict[str, str] = {}
    used: set[str] = set()
    for canonical, aliases in COLUMN_ALIASES.items():
        found = _rankings_find_column([c for c in raw.columns if c not in used], aliases)
        if found is not None:
            rename[found] = canonical
            used.add(found)
    data = raw.rename(columns=rename).copy()
    if 'player_name' not in data.columns:
        raise ValueError('No player-name column was detected.')
    if 'external_rank' not in data.columns and 'external_value' not in data.columns:
        raise ValueError('The rankings file needs a rank/ECR column or a numeric value column.')
    for col in ['external_rank', 'external_value', 'external_age', 'external_years_exp', 'source_adp', 'tier']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        else:
            data[col] = np.nan
    for col in ['position', 'nfl_team', 'sleeper_id']:
        if col not in data.columns:
            data[col] = pd.NA
    data['player_name_raw'] = data['player_name'].astype(str).str.strip()
    parsed_names = data['player_name_raw'].map(_clean_external_player_name)
    data['player_name'] = parsed_names.map(lambda item: item[0])
    parsed_teams = parsed_names.map(lambda item: item[1])
    data = data[data['player_name'].ne('') & data['player_name'].str.lower().ne('nan')].copy()
    data['normalized_name'] = data['player_name'].map(normalize_name)
    data['initial_last_key'] = data['player_name'].map(_initial_last_key)
    data['position'] = data['position'].map(normalize_position).astype('string')
    existing_team = data['nfl_team'].map(normalize_team)
    data['nfl_team'] = existing_team.where(existing_team.ne(''), parsed_teams).astype('string')
    data['sleeper_id'] = data['sleeper_id'].astype('string').str.replace('\\.0$', '', regex=True)
    if data['external_value'].notna().any():
        data['optimizer_score'] = data['external_value']
        data['score_basis'] = 'external value'
    else:
        max_rank = data['external_rank'].max(skipna=True)
        if pd.isna(max_rank):
            raise ValueError('The detected rank column contains no usable numeric values.')
        data['optimizer_score'] = float(max_rank) + 1.0 - data['external_rank']
        data['score_basis'] = 'inverse external rank'
    data = data.sort_values(['optimizer_score', 'external_rank'], ascending=[False, True], na_position='last').drop_duplicates(['normalized_name', 'position', 'nfl_team'], keep='first')
    return data.reset_index(drop=True)

def merge_rosters_with_rankings(rosters: pd.DataFrame, rankings: pd.DataFrame) -> tuple[pd.DataFrame, MatchReport]:
    left = rosters.copy().reset_index(drop=True)
    right = rankings.copy().reset_index(drop=True)
    left['normalized_name'] = left['player_name'].map(normalize_name)
    left['initial_last_key'] = left['player_name'].map(_initial_last_key)
    left['position'] = left['position'].map(normalize_position).astype('string')
    left['nfl_team'] = left.get('nfl_team', pd.Series('', index=left.index)).map(normalize_team).astype('string')
    right['position'] = right['position'].map(normalize_position).astype('string')
    right['nfl_team'] = right.get('nfl_team', pd.Series('', index=right.index)).map(normalize_team).astype('string')
    if 'initial_last_key' not in right.columns:
        right['initial_last_key'] = right['player_name'].map(_initial_last_key)

    fields = ['external_rank', 'external_value', 'external_age', 'external_years_exp', 'source_adp', 'tier', 'optimizer_score', 'score_basis']
    for col in fields:
        if col not in left.columns:
            left[col] = pd.NA
    left['_match_method'] = ''

    def apply_lookup(key_cols: list[str], method: str, *, unique_only: bool=True) -> None:
        candidates = right[right['optimizer_score'].notna()].copy()
        candidates = candidates.dropna(subset=key_cols)
        for key in key_cols:
            candidates = candidates[candidates[key].astype(str).ne('')]
        if candidates.empty:
            return
        if unique_only:
            counts = candidates.groupby(key_cols, dropna=False).size().rename('_count')
            candidates = candidates.join(counts, on=key_cols)
            candidates = candidates[candidates['_count'].eq(1)]
        candidates = candidates.sort_values('optimizer_score', ascending=False).drop_duplicates(key_cols)
        if candidates.empty:
            return
        lookup = candidates.set_index(key_cols)
        if len(key_cols) == 1:
            incoming_score = left[key_cols[0]].map(lookup['optimizer_score'])
        else:
            keys = pd.MultiIndex.from_frame(left[key_cols])
            incoming_score = pd.Series(keys.map(lookup['optimizer_score']), index=left.index)
        newly_matched = left['optimizer_score'].isna() & incoming_score.notna()
        if not newly_matched.any():
            return
        for col in fields:
            if len(key_cols) == 1:
                incoming = left[key_cols[0]].map(lookup[col])
            else:
                keys = pd.MultiIndex.from_frame(left[key_cols])
                incoming = pd.Series(keys.map(lookup[col]), index=left.index)
            mask = newly_matched & incoming.notna()
            left.loc[mask, col] = incoming.loc[mask]
        left.loc[newly_matched, '_match_method'] = method

    # Strongest to loosest. Each fallback requires a unique source candidate.
    if right['sleeper_id'].notna().any():
        by_id = right.dropna(subset=['sleeper_id']).drop_duplicates('sleeper_id').set_index('sleeper_id')
        player_ids = left['player_id'].astype('string')
        incoming_score = player_ids.map(by_id['optimizer_score'])
        newly_matched = left['optimizer_score'].isna() & incoming_score.notna()
        for col in fields:
            incoming = player_ids.map(by_id[col])
            mask = newly_matched & incoming.notna()
            left.loc[mask, col] = incoming.loc[mask]
        left.loc[newly_matched, '_match_method'] = 'Sleeper ID'

    apply_lookup(['normalized_name', 'position'], 'Exact name + position')
    apply_lookup(['normalized_name'], 'Exact name')
    apply_lookup(['initial_last_key', 'position', 'nfl_team'], 'Initial/last + position + team')
    apply_lookup(['initial_last_key', 'position'], 'Initial/last + position')

    matched_mask = pd.to_numeric(left['optimizer_score'], errors='coerce').notna()
    unmatched = tuple(left.loc[~matched_mask, 'player_name'].dropna().astype(str).tolist())
    counts = tuple((str(k), int(v)) for k, v in left.loc[matched_mask, '_match_method'].value_counts().items())
    left = left.drop(columns=['_match_method'])
    return left, MatchReport(len(left), int(matched_mask.sum()), unmatched, counts, len(right))

# ---- external_metrics.py ----
@dataclass(frozen=True)
class MetricsMatchReport:
    total_rostered: int
    matched: int
    metric_columns: tuple[str, ...]
NAME_ALIASES = ['player_name', 'player', 'name', 'player name', 'full_name', 'full name']
POSITION_ALIASES = ['position', 'pos']
TEAM_ALIASES = ['team', 'nfl team', 'nfl_team', 'tm', 'recent_team']
ID_ALIASES = ['sleeper_id', 'sleeper id', 'sleeperid']

def _metrics_norm_col(value: object) -> str:
    return ' '.join(str(value).strip().lower().replace('_', ' ').split())

def _metrics_find_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {_metrics_norm_col(col): col for col in columns}
    for alias in aliases:
        if _metrics_norm_col(alias) in normalized:
            return normalized[_metrics_norm_col(alias)]
    return None

def load_metrics_csv(file_obj: BinaryIO | TextIO) -> pd.DataFrame:
    raw = _read_uploaded_table(file_obj)
    if raw.empty:
        raise ValueError('The metrics file is empty.')
    name_col = _metrics_find_column(list(raw.columns), NAME_ALIASES)
    if name_col is None:
        raise ValueError('No player-name column was detected in the metrics file.')
    rename = {name_col: 'metrics_player_name'}
    for aliases, canonical in [(POSITION_ALIASES, 'metrics_position'), (TEAM_ALIASES, 'metrics_nfl_team'), (ID_ALIASES, 'metrics_sleeper_id')]:
        found = _metrics_find_column(list(raw.columns), aliases)
        if found and found != name_col:
            rename[found] = canonical
    data = raw.rename(columns=rename).copy()
    data['normalized_name'] = data['metrics_player_name'].map(normalize_name)
    if 'metrics_position' in data.columns:
        data['metrics_position'] = data['metrics_position'].astype('string').str.upper().str.strip()
    if 'metrics_nfl_team' in data.columns:
        data['metrics_nfl_team'] = data['metrics_nfl_team'].astype('string').str.upper().str.strip()
    if 'metrics_sleeper_id' in data.columns:
        data['metrics_sleeper_id'] = data['metrics_sleeper_id'].astype('string').str.replace('\\.0$', '', regex=True)
    protected = {'metrics_player_name', 'metrics_position', 'metrics_nfl_team', 'metrics_sleeper_id', 'normalized_name'}
    rename_metrics: dict[str, str] = {}
    existing: set[str] = set()
    for col in data.columns:
        if col in protected:
            continue
        clean = str(col).strip().lower().replace('%', ' pct ')
        clean = ''.join((ch if ch.isalnum() else '_' for ch in clean))
        clean = '_'.join((part for part in clean.split('_') if part))
        target = f"metric_{clean or 'field'}"
        suffix = 2
        while target in existing:
            target = f'metric_{clean}_{suffix}'
            suffix += 1
        existing.add(target)
        rename_metrics[col] = target
    data = data.rename(columns=rename_metrics)
    metric_cols = [col for col in data.columns if col.startswith('metric_')]
    for col in metric_cols:
        if data[col].dtype == object:
            cleaned = data[col].astype(str).str.replace(',', '', regex=False).str.replace('%', '', regex=False)
            converted = pd.to_numeric(cleaned, errors='coerce')
            if converted.notna().sum() >= max(1, int(len(data) * 0.5)):
                data[col] = converted
    id_subset = ['metrics_sleeper_id'] if 'metrics_sleeper_id' in data.columns else ['normalized_name']
    return data.drop_duplicates(id_subset, keep='last').reset_index(drop=True)

def merge_rosters_with_metrics(rosters: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, MetricsMatchReport]:
    left = rosters.copy().reset_index(drop=True)
    if 'normalized_name' not in left.columns:
        left['normalized_name'] = left['player_name'].map(normalize_name)
    metric_cols = [col for col in metrics.columns if col.startswith('metric_')]
    for col in metric_cols:
        if col not in left.columns:
            left[col] = pd.NA
    if 'metrics_sleeper_id' in metrics.columns and metrics['metrics_sleeper_id'].notna().any():
        by_id = metrics.dropna(subset=['metrics_sleeper_id']).drop_duplicates('metrics_sleeper_id')
        by_id = by_id.set_index('metrics_sleeper_id')
        id_col = 'player_id' if 'player_id' in left.columns else 'sleeper_id' if 'sleeper_id' in left.columns else None
        ids = left[id_col].astype('string') if id_col else pd.Series(pd.NA, index=left.index, dtype='string')
        for col in metric_cols:
            incoming = ids.map(by_id[col])
            mask = left[col].isna() & incoming.notna()
            left.loc[mask, col] = incoming.loc[mask]
    by_name = metrics.drop_duplicates('normalized_name').set_index('normalized_name')
    for col in metric_cols:
        incoming = left['normalized_name'].map(by_name[col])
        mask = left[col].isna() & incoming.notna()
        left.loc[mask, col] = incoming.loc[mask]
    matched = int(left[metric_cols].notna().any(axis=1).sum()) if metric_cols else 0
    return (left, MetricsMatchReport(len(left), matched, tuple(metric_cols)))

# ---- analytics.py ----
def _display_team_name(user: dict, roster_id: object) -> str:
    metadata = user.get('metadata') or {}
    return metadata.get('team_name') or metadata.get('team_name_update') or user.get('display_name') or f'Roster {roster_id}'

def build_league_roster_table(users: list[dict], rosters: list[dict], players: dict[str, dict], season: int | str | None=None) -> pd.DataFrame:
    user_map = {str(user.get('user_id')): user for user in users}
    rows: list[dict] = []
    season_num = pd.to_numeric(season, errors='coerce')
    for roster in rosters:
        owner_id = str(roster.get('owner_id')) if roster.get('owner_id') is not None else ''
        user = user_map.get(owner_id, {})
        roster_id = roster.get('roster_id')
        team_name = _display_team_name(user, roster_id)
        starters = set((str(x) for x in roster.get('starters') or []))
        reserve = set((str(x) for x in roster.get('reserve') or []))
        taxi = set((str(x) for x in roster.get('taxi') or []))
        keepers = set((str(x) for x in roster.get('keepers') or []))
        for raw_player_id in roster.get('players') or []:
            player_id = str(raw_player_id)
            player = players.get(player_id, {})
            full_name = player.get('full_name') or ' '.join((part for part in [player.get('first_name'), player.get('last_name')] if part))
            years_exp = pd.to_numeric(player.get('years_exp'), errors='coerce')
            rookie_year = pd.to_numeric(player.get('rookie_year', player.get('metadata', {}).get('rookie_year') if isinstance(player.get('metadata'), dict) else None), errors='coerce')
            second_year = bool(pd.notna(years_exp) and float(years_exp) == 1 or (pd.notna(rookie_year) and pd.notna(season_num) and (int(rookie_year) == int(season_num) - 1)))
            rows.append({'roster_id': roster_id, 'owner_id': owner_id, 'manager': user.get('display_name') or user.get('username'), 'team_name': team_name, 'player_id': player_id, 'player_name': full_name or player_id, 'normalized_name': normalize_name(full_name or player_id), 'position': player.get('position'), 'fantasy_positions': ', '.join(player.get('fantasy_positions') or []), 'nfl_team': player.get('team'), 'status': player.get('status'), 'injury_status': player.get('injury_status'), 'age': pd.to_numeric(player.get('age'), errors='coerce'), 'birth_date': player.get('birth_date'), 'rookie_year': rookie_year, 'years_exp': years_exp, 'is_second_year': second_year, 'second_year_source': 'Sleeper experience/rookie year' if second_year else 'Not identified', 'is_starter': player_id in starters, 'is_reserve': player_id in reserve, 'is_taxi': player_id in taxi, 'is_sleeper_keeper': player_id in keepers})
    columns = ['roster_id', 'owner_id', 'manager', 'team_name', 'player_id', 'player_name', 'normalized_name', 'position', 'fantasy_positions', 'nfl_team', 'status', 'injury_status', 'age', 'birth_date', 'rookie_year', 'years_exp', 'is_second_year', 'second_year_source', 'is_starter', 'is_reserve', 'is_taxi', 'is_sleeper_keeper']
    return pd.DataFrame(rows, columns=columns)

def build_standings(rosters: list[dict], users: list[dict]) -> pd.DataFrame:
    user_map = {str(user.get('user_id')): user for user in users}
    rows = []
    for roster in rosters:
        owner_id = str(roster.get('owner_id')) if roster.get('owner_id') is not None else ''
        user = user_map.get(owner_id, {})
        settings = roster.get('settings') or {}
        fpts = float(settings.get('fpts') or 0) + float(settings.get('fpts_decimal') or 0) / 100
        fpts_against = float(settings.get('fpts_against') or 0) + float(settings.get('fpts_against_decimal') or 0) / 100
        wins = int(settings.get('wins') or 0)
        losses = int(settings.get('losses') or 0)
        ties = int(settings.get('ties') or 0)
        games = wins + losses + ties
        rows.append({'team_name': _display_team_name(user, roster.get('roster_id')), 'manager': user.get('display_name') or user.get('username'), 'wins': wins, 'losses': losses, 'ties': ties, 'win_pct': round((wins + 0.5 * ties) / games, 3) if games else np.nan, 'points_for': round(fpts, 2), 'points_against': round(fpts_against, 2)})
    result = pd.DataFrame(rows)
    return result.sort_values(['wins', 'points_for'], ascending=[False, False]).reset_index(drop=True)

def calculate_team_summary(merged: pd.DataFrame, rules: KeeperRules) -> pd.DataFrame:
    work = merged.copy()
    work['optimizer_score'] = pd.to_numeric(work.get('optimizer_score'), errors='coerce')
    grouped = []
    for team_name, team in work.groupby('team_name', dropna=False):
        result = optimize_keepers(team, rules, top_n=1)
        top5 = team.nlargest(rules.keeper_count, 'optimizer_score')
        grouped.append({'team_name': team_name, 'manager': team['manager'].dropna().iloc[0] if team['manager'].notna().any() else '', 'roster_size': len(team), 'matched_players': int(team['optimizer_score'].notna().sum()), 'legal_keeper_score': round(result.total_score, 1) if result.feasible else np.nan, 'raw_top5_score': round(float(top5['optimizer_score'].sum()), 1) if top5['optimizer_score'].notna().any() else np.nan, 'rule_cost': round(result.rule_cost, 1) if result.feasible else np.nan, 'projected_qb_keepers': result.qb_count if result.feasible else np.nan, 'second_year_candidates': int(team.get('is_second_year', False).sum()), 'QB': int((team['position'] == 'QB').sum()), 'RB': int((team['position'] == 'RB').sum()), 'WR': int((team['position'] == 'WR').sum()), 'TE': int((team['position'] == 'TE').sum())})
    summary = pd.DataFrame(grouped)
    if 'legal_keeper_score' in summary.columns:
        summary = summary.sort_values('legal_keeper_score', ascending=False, na_position='last')
        summary.insert(0, 'keeper_rank', range(1, len(summary) + 1))
    return summary.reset_index(drop=True)

def project_all_keepers(merged_rosters: pd.DataFrame, rules: KeeperRules) -> tuple[pd.DataFrame, dict[str, KeeperResult]]:
    keeper_frames: list[pd.DataFrame] = []
    results: dict[str, KeeperResult] = {}
    for team_name, team in merged_rosters.groupby('team_name'):
        result = optimize_keepers(team, rules, top_n=3)
        results[str(team_name)] = result
        if not result.feasible:
            continue
        selected = result.keepers.copy()
        selected['team_name'] = team_name
        selected['keeper_slot'] = range(1, len(selected) + 1)
        keeper_frames.append(selected)
    if not keeper_frames:
        return (pd.DataFrame(), results)
    projected = pd.concat(keeper_frames, ignore_index=True)
    return (projected.sort_values(['team_name', 'optimizer_score'], ascending=[True, False]), results)

def find_best_available(merged_rosters: pd.DataFrame, rankings: pd.DataFrame, rules: KeeperRules, drafted_player_ids: Iterable[str] | None=None, drafted_names: Iterable[str] | None=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    projected_keepers, _ = project_all_keepers(merged_rosters, rules)
    keeper_ids = set(projected_keepers.get('player_id', pd.Series(dtype=str)).dropna().astype(str))
    keeper_names = set(projected_keepers.get('normalized_name', pd.Series(dtype=str)).dropna().astype(str))
    drafted_ids = set((str(x) for x in drafted_player_ids or []))
    drafted_normalized = set((normalize_name(x) for x in drafted_names or []))
    available = rankings.copy()
    if 'sleeper_id' in available.columns:
        available = available[~available['sleeper_id'].astype('string').isin(keeper_ids | drafted_ids)]
    available = available[~available['normalized_name'].isin(keeper_names | drafted_normalized)].copy()
    available = available.sort_values(['optimizer_score', 'external_rank'], ascending=[False, True], na_position='last').reset_index(drop=True)
    available.insert(0, 'board_rank', range(1, len(available) + 1))
    return (available, projected_keepers.reset_index(drop=True))

def position_scarcity(available: pd.DataFrame, cutoffs: tuple[int, ...]=(25, 50, 100)) -> pd.DataFrame:
    rows: list[dict] = []
    for position in CORE_POSITIONS:
        pos = available[available['position'] == position].copy()
        row = {'position': position, 'total_available': len(pos)}
        for cutoff in cutoffs:
            row[f'in_top_{cutoff}'] = int((available.head(cutoff)['position'] == position).sum())
        row['best_board_rank'] = int(pos['board_rank'].min()) if not pos.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def build_draft_picks_table(picks: list[dict], players: dict[str, dict], users: list[dict] | None=None) -> pd.DataFrame:
    user_map = {str(user.get('user_id')): user for user in users or []}
    rows: list[dict] = []
    for pick in picks:
        player_id = str(pick.get('player_id') or '')
        player = players.get(player_id, {})
        metadata = pick.get('metadata') or {}
        picked_by = str(pick.get('picked_by') or '')
        user = user_map.get(picked_by, {})
        player_name = (player.get('full_name') or metadata.get('first_name', '') + ' ' + metadata.get('last_name', '')).strip()
        rows.append({'pick_no': pick.get('pick_no'), 'round': pick.get('round'), 'draft_slot': pick.get('draft_slot'), 'roster_id': pick.get('roster_id'), 'picked_by': picked_by, 'manager': user.get('display_name') or user.get('username') or picked_by, 'player_id': player_id, 'player_name': player_name or player_id, 'position': player.get('position') or metadata.get('position'), 'nfl_team': player.get('team') or metadata.get('team'), 'timestamp': pick.get('picked_at')})
    table = pd.DataFrame(rows)
    if not table.empty and 'pick_no' in table.columns:
        table = table.sort_values('pick_no').reset_index(drop=True)
    return table

def roster_position_counts(team_df: pd.DataFrame) -> dict[str, int]:
    counter = Counter(team_df.get('position', pd.Series(dtype=str)).dropna().astype(str))
    return {position: int(counter.get(position, 0)) for position in CORE_POSITIONS}

def summarize_alternative(option_df: pd.DataFrame) -> str:
    return ', '.join(option_df['player_name'].astype(str).tolist())

# ---- questions.py ----
def answer_league_question(question: str, roster_data: pd.DataFrame, team_summary: pd.DataFrame, available: pd.DataFrame | None=None, projected_keepers: pd.DataFrame | None=None) -> str:
    q = question.strip().lower()
    if not q:
        return 'Enter a question about the league, keepers, or projected draft pool.'
    if any((term in q for term in ['best keeper', 'strongest keeper', 'best team', 'strongest team'])):
        valid = team_summary.dropna(subset=['legal_keeper_score'])
        if valid.empty:
            return 'Upload rankings first so legal keeper strength can be calculated.'
        row = valid.iloc[0]
        return f"{row['team_name']} has the strongest projected legal keeper core with a score of {row['legal_keeper_score']:,.1f}."
    if any((term in q for term in ['worst keeper', 'weakest keeper', 'weakest team'])):
        valid = team_summary.dropna(subset=['legal_keeper_score'])
        if valid.empty:
            return 'Upload rankings first so legal keeper strength can be calculated.'
        row = valid.iloc[-1]
        return f"{row['team_name']} has the weakest projected legal keeper core with a score of {row['legal_keeper_score']:,.1f}."
    if 'how many' in q and 'qb' in q and ('keep' in q):
        if projected_keepers is None or projected_keepers.empty:
            return 'Upload rankings first so projected keepers can be calculated.'
        count = int((projected_keepers['position'] == 'QB').sum())
        names = projected_keepers.loc[projected_keepers['position'] == 'QB', 'player_name'].astype(str).tolist()
        return f"The model projects {count} quarterback keepers: {', '.join(names) or 'none'}."
    position_match = re.search('\\b(qb|rb|wr|te)\\b', q)
    if available is not None and (not available.empty) and any((term in q for term in ['best available', 'top available', 'draft'])):
        pool = available
        if position_match:
            position = position_match.group(1).upper()
            pool = pool[pool['position'] == position]
        if pool.empty:
            return 'No matching players were found in the projected available pool.'
        top = pool.head(5)
        return 'Top projected available players: ' + '; '.join((f'{row.player_name} ({row.position}, board {int(row.board_rank)})' for row in top.itertuples()))
    if available is not None and (not available.empty) and ('scarcity' in q):
        scarcity = position_scarcity(available)
        thin = scarcity.sort_values('in_top_50').iloc[0]
        return f"{thin['position']} is the thinnest core position in the top 50 projected available players, with {int(thin['in_top_50'])} options."
    normalized_q = normalize_name(question)
    candidates = roster_data.copy()
    candidates['_match'] = candidates['normalized_name'].map(lambda name: name in normalized_q or normalized_q in name)
    matches = candidates[candidates['_match'] & candidates['optimizer_score'].notna()]
    if len(matches) == 1:
        row = matches.iloc[0]
        return f"{row['player_name']} is on {row['team_name']}, ranked {row.get('external_rank', '—')}, with source value {row.get('external_value', '—')}. Second-year eligible: {('yes' if bool(row.get('is_second_year')) else 'no')}."
    return 'I could not map that wording to a built-in answer yet. Try asking who has the strongest keeper core, how many QBs will be kept, the best available RB/WR/QB/TE, or which position is scarce.'

# ---- sleeper_client.py ----
class SleeperAPIError(RuntimeError):
    """Raised when Sleeper data cannot be fetched or parsed."""

class SleeperClient:
    BASE_URL = 'https://api.sleeper.app/v1'

    def __init__(self, timeout_seconds: int=25) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        retry = Retry(total=3, connect=3, read=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=('GET',))
        self.session.mount('https://', HTTPAdapter(max_retries=retry))
        self.session.headers.update({'User-Agent': 'BFTD-Keeper-Lab/2.0'})

    def _get(self, path: str) -> Any:
        url = f'{self.BASE_URL}{path}'
        try:
            response = self.session.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise SleeperAPIError(f'Sleeper request failed for {url}: {exc}') from exc
        except ValueError as exc:
            raise SleeperAPIError(f'Sleeper returned invalid JSON for {url}.') from exc

    def get_league(self, league_id: str) -> dict[str, Any]:
        data = self._get(f'/league/{league_id}')
        if not data:
            raise SleeperAPIError(f'Sleeper league {league_id} was not found.')
        return data

    def get_users(self, league_id: str) -> list[dict[str, Any]]:
        return self._get(f'/league/{league_id}/users') or []

    def get_rosters(self, league_id: str) -> list[dict[str, Any]]:
        return self._get(f'/league/{league_id}/rosters') or []

    def get_league_drafts(self, league_id: str) -> list[dict[str, Any]]:
        return self._get(f'/league/{league_id}/drafts') or []

    def get_draft(self, draft_id: str) -> dict[str, Any]:
        return self._get(f'/draft/{draft_id}') or {}

    def get_draft_picks(self, draft_id: str) -> list[dict[str, Any]]:
        return self._get(f'/draft/{draft_id}/picks') or []

    def get_draft_traded_picks(self, draft_id: str) -> list[dict[str, Any]]:
        return self._get(f'/draft/{draft_id}/traded_picks') or []

    def get_matchups(self, league_id: str, week: int) -> list[dict[str, Any]]:
        return self._get(f'/league/{league_id}/matchups/{week}') or []

    def get_transactions(self, league_id: str, week: int) -> list[dict[str, Any]]:
        return self._get(f'/league/{league_id}/transactions/{week}') or []

    def get_traded_picks(self, league_id: str) -> list[dict[str, Any]]:
        return self._get(f'/league/{league_id}/traded_picks') or []

    def get_players_nfl(self) -> dict[str, dict[str, Any]]:
        return self._get('/players/nfl') or {}

    def get_nfl_state(self) -> dict[str, Any]:
        return self._get('/state/nfl') or {}

# ---- Streamlit application ----
st.set_page_config(page_title=APP_NAME, page_icon='🏈', layout='wide', initial_sidebar_state='expanded')
st.markdown('\n    <style>\n      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}\n      [data-testid="stMetricValue"] {font-size: 1.55rem;}\n      .keeper-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: .8rem;}\n      @media (max-width: 700px) {\n        .block-container {padding-left: .7rem; padding-right: .7rem;}\n        h1 {font-size: 1.7rem !important;}\n        h2 {font-size: 1.35rem !important;}\n      }\n    </style>\n    ', unsafe_allow_html=True)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_league_core(league_id: str):
    client = SleeperClient()
    league = client.get_league(league_id)
    users = client.get_users(league_id)
    rosters = client.get_rosters(league_id)
    drafts = client.get_league_drafts(league_id)
    nfl_state = client.get_nfl_state()
    return (league, users, rosters, drafts, nfl_state)

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_players():
    return SleeperClient().get_players_nfl()

@st.cache_data(ttl=3, show_spinner=False)
def fetch_live_draft(draft_id: str):
    client = SleeperClient(timeout_seconds=15)
    return (client.get_draft(draft_id), client.get_draft_picks(draft_id))

def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode('utf-8')

def apply_second_year_overrides(data: pd.DataFrame, override_file) -> tuple[pd.DataFrame, int]:
    if override_file is None:
        return (data, 0)
    raw = _read_uploaded_table(override_file)
    normalized_cols = {str(c).strip().lower().replace('_', ' '): c for c in raw.columns}
    name_col = normalized_cols.get('player name') or normalized_cols.get('player') or normalized_cols.get('name')
    flag_col = normalized_cols.get('is second year') or normalized_cols.get('second year') or normalized_cols.get('eligible')
    if name_col is None or flag_col is None:
        raise ValueError('The override file needs player_name and is_second_year columns.')
    flags = raw[[name_col, flag_col]].copy()
    flags['normalized_name'] = flags[name_col].map(normalize_name)
    flags['override'] = flags[flag_col].astype(str).str.strip().str.lower().isin({'true', '1', 'yes', 'y'})
    mapping = flags.drop_duplicates('normalized_name').set_index('normalized_name')['override']
    result = data.copy()
    mask = result['normalized_name'].isin(mapping.index)
    result.loc[mask, 'is_second_year'] = result.loc[mask, 'normalized_name'].map(mapping)
    result.loc[mask, 'second_year_source'] = 'Manual override'
    return (result, int(mask.sum()))

def safe_columns(frame: pd.DataFrame, requested: list[str]) -> list[str]:
    return [col for col in requested if col in frame.columns]
st.title('🏈 BFTD Keeper Lab')
st.caption(f'Keeper-league analytics and synchronized Sleeper draft board · version {APP_VERSION}')
with st.sidebar:
    st.header('League setup')
    league_id = st.text_input('Sleeper league ID', value=DEFAULT_LEAGUE_ID).strip()
    st.markdown(f'**Locked rules**  \nExactly **{KEEPER_RULES.keeper_count}** keepers  \nMaximum **{KEEPER_RULES.max_qb} QB** keeper  \nAt least **{KEEPER_RULES.min_second_year} second-year** player')
    refresh = st.button('Refresh Sleeper data', use_container_width=True, type='primary')
    st.caption('Sleeper access is read-only. No password or token is required.')
if not league_id:
    st.error('Enter a Sleeper league ID.')
    st.stop()
if refresh:
    fetch_league_core.clear()
    fetch_live_draft.clear()
try:
    with st.spinner('Syncing league data from Sleeper…'):
        league, users, rosters, drafts, nfl_state = fetch_league_core(league_id)
        players = fetch_players()
except SleeperAPIError as exc:
    st.error(str(exc))
    st.info('Check the league ID and internet connection, then tap **Refresh Sleeper data**.')
    st.stop()
league_name = league.get('name', 'Sleeper League')
season = league.get('season', nfl_state.get('league_season', '—'))
roster_table = build_league_roster_table(users, rosters, players, season=season)
standings = build_standings(rosters, users)
if roster_table.empty:
    st.warning('Sleeper returned no rostered players for this league.')
with st.expander('Data sources and eligibility', expanded=True):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        rankings_file = st.file_uploader('1. Rankings/value file', help='Select CSV, Excel, or a ZIP containing one. The unrestricted picker fixes Android files that appear greyed out.')
    with col_b:
        metrics_files = st.file_uploader('2. Advanced metrics files', accept_multiple_files=True, help='Optional CSV, Excel, or ZIP exports such as usage, efficiency, projections, or injury metrics.')
    with col_c:
        overrides_file = st.file_uploader('3. Second-year overrides', help='Optional CSV or Excel file with player_name and is_second_year columns.')
try:
    roster_table, override_count = apply_second_year_overrides(roster_table, overrides_file)
    if override_count:
        st.success(f'Applied second-year overrides to {override_count} rostered players.')
except (ValueError, pd.errors.ParserError) as exc:
    st.error(f'Could not use the eligibility override file: {exc}')
rankings = pd.DataFrame()
rankings_enriched = pd.DataFrame()
merged = roster_table.copy()
match_report = None
metrics_reports = []
if rankings_file is not None:
    try:
        rankings = load_rankings_csv(rankings_file)
        merged, match_report = merge_rosters_with_rankings(roster_table, rankings)
        rankings_enriched = rankings.copy()
        non_defense = merged['position'].ne('DEF')
        non_defense_total = int(non_defense.sum())
        non_defense_matched = int(merged.loc[non_defense, 'optimizer_score'].notna().sum())
        unmatched_non_defense = merged.loc[non_defense & merged['optimizer_score'].isna(), 'player_name'].dropna().astype(str).tolist()
        unmatched_defenses = merged.loc[merged['position'].eq('DEF') & merged['optimizer_score'].isna(), 'player_name'].dropna().astype(str).tolist()
        st.success(f'Rankings loaded: {len(rankings):,} players; matched {non_defense_matched:,} of {non_defense_total:,} non-defense rostered players.')
        if unmatched_defenses:
            st.caption(f'{len(unmatched_defenses)} team defenses are not in this dynasty/keeper ranking file and are excluded from the player match rate.')
        if len(rankings) < 100:
            st.warning('This rankings file contains fewer than 100 usable player rows. It may be a partial export rather than the full rankings list.')
        if non_defense_total and non_defense_matched / non_defense_total < 0.7:
            st.warning('The match rate is still low. Open the unmatched list and verify that the uploaded file is the full NFL rankings export, not a filtered or partial table.')
        if match_report.method_counts:
            method_text = ' · '.join(f'{method}: {count}' for method, count in match_report.method_counts)
            st.caption(f'Match methods — {method_text}')
        if unmatched_non_defense:
            with st.expander(f'Unmatched non-defense players ({len(unmatched_non_defense)})'):
                st.write(', '.join(unmatched_non_defense))
        if unmatched_defenses:
            with st.expander(f'Unranked team defenses ({len(unmatched_defenses)})'):
                st.write(', '.join(unmatched_defenses))
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f'Could not use the rankings file: {exc}')
for metric_file in metrics_files or []:
    try:
        metrics = load_metrics_csv(metric_file)
        merged, report = merge_rosters_with_metrics(merged, metrics)
        if not rankings_enriched.empty:
            rankings_enriched, _ = merge_rosters_with_metrics(rankings_enriched, metrics)
        metrics_reports.append((metric_file.name, report))
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f'Could not use {metric_file.name}: {exc}')
if metrics_reports:
    total_metric_cols = len({c for _, report in metrics_reports for c in report.metric_columns})
    st.success(f'Loaded {len(metrics_reports)} metrics file(s), adding {total_metric_cols} searchable metric columns.')
m1, m2, m3, m4 = st.columns(4)
m1.metric('League', league_name)
m2.metric('Season', season)
m3.metric('Teams', len(rosters))
m4.metric('Sleeper status', str(league.get('status', '—')).replace('_', ' ').title())
team_names = sorted(roster_table['team_name'].dropna().astype(str).unique().tolist())
rankings_for_pool = rankings_enriched if not rankings_enriched.empty else rankings
summary = calculate_team_summary(merged, KEEPER_RULES) if not rankings.empty else pd.DataFrame()
if not rankings.empty:
    projected_available, projected_keepers = find_best_available(merged, rankings_for_pool, KEEPER_RULES)
else:
    projected_available, projected_keepers = (pd.DataFrame(), pd.DataFrame())
tab_dashboard, tab_keepers, tab_pool, tab_draft, tab_compare, tab_ask, tab_data = st.tabs(['Dashboard', 'Keepers', 'Draft pool', 'Live draft', 'Compare', 'Ask league', 'Data'])
with tab_dashboard:
    st.subheader('League dashboard')
    if not standings.empty:
        st.markdown('#### Sleeper standings')
        st.dataframe(standings, use_container_width=True, hide_index=True)
    if rankings.empty:
        st.info('Upload a rankings or value CSV to unlock legal keeper rankings and the projected draft pool.')
    else:
        st.markdown('#### Legal keeper strength')
        st.dataframe(summary, use_container_width=True, hide_index=True)
        best = summary.dropna(subset=['legal_keeper_score']).head(1)
        if not best.empty:
            row = best.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric('Strongest keeper core', str(row['team_name']))
            c2.metric('Projected QB keepers', int(projected_keepers['position'].eq('QB').sum()))
            c3.metric('Players entering draft', len(projected_available))
        st.markdown('#### Projected keepers by position')
        keeper_counts = projected_keepers['position'].value_counts().rename_axis('position').reset_index(name='keepers') if not projected_keepers.empty else pd.DataFrame()
        st.dataframe(keeper_counts, use_container_width=True, hide_index=True)
with tab_keepers:
    st.subheader('Rule-aware keeper optimizer')
    selected_team = st.selectbox('Team', team_names, key='keeper_team') if team_names else None
    if selected_team:
        team_df = merged.loc[merged['team_name'] == selected_team].copy()
        display_cols = safe_columns(team_df, ['player_name', 'position', 'nfl_team', 'external_rank', 'external_value', 'source_adp', 'tier', 'age', 'years_exp', 'is_second_year', 'injury_status'])
        editor = team_df[display_cols].copy()
        editor['locked'] = False
        editor['excluded'] = False
        edited = st.data_editor(editor, use_container_width=True, hide_index=True, disabled=[c for c in editor.columns if c not in {'is_second_year', 'locked', 'excluded'}], column_config={'is_second_year': st.column_config.CheckboxColumn('Second-year eligible'), 'locked': st.column_config.CheckboxColumn('Lock'), 'excluded': st.column_config.CheckboxColumn('Exclude')}, key=f'keeper_editor_{selected_team}')
        calculation_df = team_df.copy()
        for col in ['is_second_year', 'locked', 'excluded']:
            calculation_df[col] = edited[col].values
        if rankings.empty:
            st.warning('Upload rankings/value data to optimize this keeper group.')
        else:
            result = optimize_keepers(calculation_df, KEEPER_RULES, top_n=6)
            if not result.feasible:
                st.error(result.explanation)
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('Legal keeper score', f'{result.total_score:,.1f}')
                c2.metric('QB keepers', result.qb_count)
                c3.metric('Second-year keepers', result.second_year_count)
                c4.metric('Cost of rules', f'{result.rule_cost:,.1f}')
                keeper_cols = safe_columns(result.keepers, ['player_name', 'position', 'nfl_team', 'external_rank', 'external_value', 'source_adp', 'tier', 'age', 'years_exp', 'is_second_year', 'optimizer_score', 'injury_status'])
                st.markdown('#### Recommended five')
                st.dataframe(result.keepers[keeper_cols], use_container_width=True, hide_index=True)
                st.caption(result.explanation)
                st.download_button('Download recommended keepers', to_csv_bytes(result.keepers[keeper_cols]), file_name=f'{selected_team}_recommended_keepers.csv', mime='text/csv')
                if result.alternatives:
                    st.markdown('#### Best legal alternatives')
                    alt_rows = []
                    for number, option in enumerate(result.alternatives, start=2):
                        alt_rows.append({'option': number, 'score': round(option.total_score, 1), 'difference_from_best': round(result.total_score - option.total_score, 1), 'QB': option.qb_count, 'second_year': option.second_year_count, 'keepers': summarize_alternative(option.keepers)})
                    st.dataframe(pd.DataFrame(alt_rows), use_container_width=True, hide_index=True)
with tab_pool:
    st.subheader('Projected post-keeper draft pool')
    if rankings.empty:
        st.info('Upload rankings/value data to generate the draft pool.')
    else:
        filter_col1, filter_col2 = st.columns(2)
        positions = sorted(projected_available['position'].dropna().astype(str).unique().tolist())
        position_filter = filter_col1.multiselect('Positions', positions)
        top_n = filter_col2.slider('Show top', 10, min(300, max(10, len(projected_available))), min(100, max(10, len(projected_available))), 10)
        pool_view = projected_available.copy()
        if position_filter:
            pool_view = pool_view[pool_view['position'].isin(position_filter)]
        board_cols = safe_columns(pool_view, ['board_rank', 'player_name', 'position', 'nfl_team', 'external_rank', 'external_value', 'source_adp', 'tier', 'external_age', 'external_years_exp'] + [c for c in pool_view.columns if c.startswith('metric_')][:8])
        st.dataframe(pool_view.head(top_n)[board_cols], use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('#### Position scarcity')
            st.dataframe(position_scarcity(projected_available), use_container_width=True, hide_index=True)
        with c2:
            st.markdown('#### Projected keepers')
            projected_cols = safe_columns(projected_keepers, ['team_name', 'player_name', 'position', 'nfl_team', 'external_rank', 'external_value', 'is_second_year'])
            st.dataframe(projected_keepers[projected_cols], use_container_width=True, hide_index=True, height=350)
        d1, d2 = st.columns(2)
        d1.download_button('Download full draft board', to_csv_bytes(projected_available), file_name='bftd_projected_draft_board.csv', mime='text/csv', use_container_width=True)
        d2.download_button('Download projected keepers', to_csv_bytes(projected_keepers), file_name='bftd_projected_keepers.csv', mime='text/csv', use_container_width=True)
with tab_draft:
    st.subheader('Synchronized Sleeper draft assistant')
    if not drafts:
        st.info('No draft objects are currently attached to this Sleeper league.')
    else:
        draft_labels = {str(d.get('draft_id')): f"{d.get('season', '')} · {d.get('status', 'unknown').title()} · {d.get('type', 'draft').title()}" for d in drafts}
        selected_draft_id = st.selectbox('Draft', list(draft_labels), format_func=lambda draft_id: draft_labels[draft_id])
        auto_refresh = st.toggle('Auto-refresh every 10 seconds', value=False)
        user_team = st.selectbox('Your team', team_names, key='draft_team') if team_names else None
        if st.button('Refresh draft now', use_container_width=True):
            fetch_live_draft.clear()

        def render_live_panel():
            try:
                draft_detail, raw_picks = fetch_live_draft(selected_draft_id)
            except SleeperAPIError as exc:
                st.error(str(exc))
                return
            picks_table = build_draft_picks_table(raw_picks, players, users)
            if not picks_table.empty:
                roster_name_map = roster_table[['roster_id', 'team_name']].drop_duplicates('roster_id').set_index('roster_id')['team_name']
                picks_table['team_name'] = picks_table['roster_id'].map(roster_name_map)
            total_rounds = int((draft_detail.get('settings') or {}).get('rounds') or 0)
            total_teams = int((draft_detail.get('settings') or {}).get('teams') or len(rosters) or 0)
            total_picks = total_rounds * total_teams if total_rounds and total_teams else None
            p1, p2, p3, p4 = st.columns(4)
            p1.metric('Draft status', str(draft_detail.get('status', '—')).title())
            p2.metric('Picks made', len(picks_table))
            p3.metric('Current/next pick', len(picks_table) + 1)
            p4.metric('Picks remaining', max(0, total_picks - len(picks_table)) if total_picks else '—')
            if not picks_table.empty:
                st.markdown('#### Latest picks')
                st.dataframe(picks_table.sort_values('pick_no', ascending=False).head(12), use_container_width=True, hide_index=True)
            if rankings.empty:
                st.warning('Upload rankings to display the live remaining board.')
                return
            drafted_ids = picks_table.get('player_id', pd.Series(dtype=str)).dropna().astype(str).tolist()
            drafted_names = picks_table.get('player_name', pd.Series(dtype=str)).dropna().astype(str).tolist()
            live_available, _ = find_best_available(merged, rankings_for_pool, KEEPER_RULES, drafted_player_ids=drafted_ids, drafted_names=drafted_names)
            st.markdown('#### Best players still available')
            live_cols = safe_columns(live_available, ['board_rank', 'player_name', 'position', 'nfl_team', 'external_rank', 'external_value', 'source_adp', 'tier'])
            st.dataframe(live_available.head(25)[live_cols], use_container_width=True, hide_index=True)
            if user_team:
                keeper_team = projected_keepers[projected_keepers['team_name'] == user_team]
                drafted_to_user = picks_table[picks_table['team_name'].eq(user_team)] if 'team_name' in picks_table.columns else pd.DataFrame()
                combined = pd.concat([keeper_team[['position']] if 'position' in keeper_team else pd.DataFrame(), drafted_to_user[['position']] if 'position' in drafted_to_user else pd.DataFrame()], ignore_index=True)
                counts = roster_position_counts(combined)
                st.markdown('#### Your keeper + drafted position counts')
                st.dataframe(pd.DataFrame([counts]), use_container_width=True, hide_index=True)
        if hasattr(st, 'fragment'):
            interval = '10s' if auto_refresh else None

            @st.fragment(run_every=interval)
            def live_fragment():
                render_live_panel()
            live_fragment()
        else:
            if st.button('Refresh draft picks'):
                fetch_live_draft.clear()
            render_live_panel()
with tab_compare:
    st.subheader('Player comparison')
    player_options = sorted(merged['player_name'].dropna().astype(str).unique().tolist())
    if len(player_options) < 2:
        st.info('At least two rostered players are required.')
    else:
        c1, c2 = st.columns(2)
        player_a = c1.selectbox('Player A', player_options, index=0)
        player_b = c2.selectbox('Player B', player_options, index=1)
        compare = merged[merged['player_name'].isin([player_a, player_b])].copy()
        compare_cols = safe_columns(compare, ['player_name', 'team_name', 'position', 'nfl_team', 'age', 'years_exp', 'is_second_year', 'external_rank', 'external_value', 'source_adp', 'tier', 'optimizer_score', 'injury_status'] + [c for c in compare.columns if c.startswith('metric_')])
        st.dataframe(compare[compare_cols], use_container_width=True, hide_index=True)
        numeric_cols = [col for col in compare_cols if col not in {'player_name', 'team_name'} and pd.api.types.is_numeric_dtype(compare[col])]
        selected_metrics = st.multiselect('Chart metrics', numeric_cols, default=numeric_cols[:5])
        if selected_metrics:
            st.bar_chart(compare.set_index('player_name')[selected_metrics].T)
with tab_ask:
    st.subheader('Ask the league')
    st.caption('Deterministic answers grounded in the synced league and uploaded rankings—no external AI key required.')
    question = st.text_input('Question', placeholder='Who has the strongest keeper core?')
    if st.button('Answer', type='primary'):
        response = answer_league_question(question, merged, summary, projected_available if not projected_available.empty else None, projected_keepers if not projected_keepers.empty else None)
        st.info(response)
    st.markdown('Try: **Who has the strongest keeper core?** · **How many QBs will be kept?** · **Who are the best available RBs?** · **Which position is scarce?**')
with tab_data:
    st.subheader('League data explorer')
    f1, f2 = st.columns(2)
    team_filter = f1.multiselect('Teams', team_names)
    position_values = sorted(merged['position'].dropna().astype(str).unique().tolist())
    position_filter = f2.multiselect('Positions', position_values)
    view = merged.copy()
    if team_filter:
        view = view[view['team_name'].isin(team_filter)]
    if position_filter:
        view = view[view['position'].isin(position_filter)]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button('Download merged league dataset', to_csv_bytes(view), file_name='bftd_merged_league_data.csv', mime='text/csv')
    with st.expander('Sleeper league configuration'):
        st.json({'league_id': league_id, 'roster_positions': league.get('roster_positions', []), 'scoring_settings': league.get('scoring_settings', {}), 'settings': league.get('settings', {}), 'drafts': drafts, 'nfl_state': nfl_state})

