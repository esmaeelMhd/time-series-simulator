import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, strategies as st

from timesim.data.schema import VariableSchema


@given(
    st.lists(
        st.text(min_size=1, max_size=8).filter(lambda s: s.isidentifier()),
        min_size=1,
        max_size=12,
        unique=True,
    )
)
def test_variable_schema_roundtrip_groups(columns):
    # Partition columns into roles while preserving uniqueness.
    c_split = max(1, len(columns) // 3)
    x_split = max(c_split + 1, (2 * len(columns)) // 3)
    groups = {
        "control": columns[:c_split],
        "exogenous": columns[c_split:x_split],
        "objective": columns[x_split:],
    }
    schema = VariableSchema.from_groups(groups)
    assert schema.to_groups() == groups
    schema.validate_columns(columns, require_exact_match=True)


@given(
    st.text(min_size=1, max_size=8).filter(lambda s: s.isidentifier()),
)
def test_variable_schema_rejects_duplicate_column(col):
    groups = {
        "control": [col],
        "exogenous": [col],
        "objective": [],
    }
    with pytest.raises(ValueError):
        VariableSchema.from_groups(groups)
