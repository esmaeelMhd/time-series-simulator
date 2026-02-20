from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Sequence


class VariableRole(str, Enum):
    """Canonical variable roles for world-model datasets."""

    CONTROL = "control"
    EXOGENOUS = "exogenous"
    OBJECTIVE = "objective"


_ALLOWED_ROLE_NAMES = {r.value for r in VariableRole}


@dataclass(frozen=True)
class VariableSchema:
    """Single source of truth for dataset variable taxonomy."""

    role_by_column: Dict[str, VariableRole]
    ordered_columns: List[str]

    @classmethod
    def from_groups(cls, groups: Mapping[str, Sequence[str]]) -> "VariableSchema":
        role_by_column: Dict[str, VariableRole] = {}
        ordered_columns: List[str] = []

        unknown_group_keys = [k for k in groups.keys() if k not in _ALLOWED_ROLE_NAMES]
        if unknown_group_keys:
            raise ValueError(
                "Variable groups contain unsupported keys: "
                f"{unknown_group_keys}. Allowed: {sorted(_ALLOWED_ROLE_NAMES)}"
            )

        for role_name in (VariableRole.CONTROL.value, VariableRole.EXOGENOUS.value, VariableRole.OBJECTIVE.value):
            cols = list(groups.get(role_name, []))
            for col in cols:
                if col in role_by_column:
                    prev = role_by_column[col].value
                    raise ValueError(
                        f"Column '{col}' is assigned to both '{prev}' and '{role_name}'. "
                        "Each column must map to exactly one role."
                    )
                role_by_column[col] = VariableRole(role_name)
                ordered_columns.append(col)

        if not role_by_column:
            raise ValueError("Variable schema is empty. Provide at least one mapped column.")

        return cls(role_by_column=role_by_column, ordered_columns=ordered_columns)

    @classmethod
    def from_column_roles(cls, column_roles: Mapping[str, str]) -> "VariableSchema":
        role_by_column: Dict[str, VariableRole] = {}
        ordered_columns: List[str] = []

        for col, role_name in column_roles.items():
            if role_name not in _ALLOWED_ROLE_NAMES:
                raise ValueError(
                    f"Invalid role '{role_name}' for column '{col}'. "
                    f"Allowed roles: {sorted(_ALLOWED_ROLE_NAMES)}"
                )
            if col in role_by_column:
                raise ValueError(f"Duplicate column in variable role mapping: '{col}'")
            role_by_column[col] = VariableRole(role_name)
            ordered_columns.append(col)

        if not role_by_column:
            raise ValueError("Variable schema is empty. Provide at least one mapped column.")

        return cls(role_by_column=role_by_column, ordered_columns=ordered_columns)

    def to_groups(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {
            VariableRole.CONTROL.value: [],
            VariableRole.EXOGENOUS.value: [],
            VariableRole.OBJECTIVE.value: [],
        }
        for col in self.ordered_columns:
            groups[self.role_by_column[col].value].append(col)
        return groups

    def columns_for_role(self, role: VariableRole) -> List[str]:
        return [col for col in self.ordered_columns if self.role_by_column[col] == role]

    def columns_for_group_names(self, group_names: Iterable[str]) -> List[str]:
        requested_roles: List[VariableRole] = []
        for group in group_names:
            if group not in _ALLOWED_ROLE_NAMES:
                raise ValueError(
                    f"Unknown variable group '{group}'. Allowed groups: {sorted(_ALLOWED_ROLE_NAMES)}"
                )
            requested_roles.append(VariableRole(group))

        requested_set = set(requested_roles)
        return [col for col in self.ordered_columns if self.role_by_column[col] in requested_set]

    def role_for(self, column: str) -> VariableRole:
        if column not in self.role_by_column:
            raise KeyError(f"Column '{column}' is not present in the variable schema")
        return self.role_by_column[column]

    def validate_columns(self, columns: Sequence[str], *, require_exact_match: bool = True) -> None:
        """Validate schema coverage against DataFrame columns.

        When ``require_exact_match=True``, each input column must be assigned to
        exactly one of {control, exogenous, objective}, and schema columns must
        all exist in the provided column list.
        """
        columns_set = set(columns)
        mapped_set = set(self.role_by_column.keys())

        missing = [c for c in self.ordered_columns if c not in columns_set]
        if missing:
            raise ValueError(
                f"Variable schema references columns missing from dataset: {missing}"
            )

        if require_exact_match:
            unmapped = [c for c in columns if c not in mapped_set]
            if unmapped:
                raise ValueError(
                    "Dataset columns without variable role assignment: "
                    f"{unmapped}. Map every dataset column to exactly one of "
                    f"{sorted(_ALLOWED_ROLE_NAMES)}."
                )
