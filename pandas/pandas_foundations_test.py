
# pandas_foundations_test.py
# Foundations Test: 10 coding questions based on "10 Minutes to pandas"
# Instructions:
# 1) Implement the functions marked with "TODO".
# 2) Run this file: `python pandas_foundations_test.py`
# 3) All asserts should pass. Do not change the provided data.

import pandas as pd
import numpy as np

# ---------- Test Data ----------

# Customers table
customers = pd.DataFrame({
    "customer_id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Cathy", "Dan", "Eva"],
    "city": ["Dubai", "Dubai", "Abu Dhabi", "Sharjah", "Dubai"],
    "signup": pd.to_datetime(["2025-01-04", "2025-01-06", "2025-02-01", "2025-02-14", "2025-03-01"]),
})

# Orders table
orders = pd.DataFrame({
    "order_id": [100, 101, 102, 103, 104, 105, 106],
    "customer_id": [1, 1, 2, 3, 3, 3, 5],
    "amount": [120.0, np.nan, 50.0, 75.0, 0.0, 200.0, 130.0],
    "status": ["paid", "paid", "refund", "paid", "paid", "paid", "paid"],
    "created_at": pd.to_datetime(["2025-03-02", "2025-03-05", "2025-03-05", "2025-03-07", "2025-03-07", "2025-03-10", "2025-03-12"]),
})

# Products table
products = pd.DataFrame({
    "sku": ["A1", "A2", "B1", "B2"],
    "category": ["supplement", "supplement", "equipment", "equipment"],
    "price": [25.0, 40.0, 120.0, 220.0],
})

# Line items (many-to-many orders-products)
line_items = pd.DataFrame({
    "order_id": [100, 100, 101, 102, 103, 104, 105, 105, 106],
    "sku":      ["A1", "B1", "A2", "A1", "A1", "B2", "A2", "B1", "A1"],
    "qty":      [2,     1,    1,    3,    2,    1,    1,    1,    4],
})

# A messy frame to clean
messy = pd.DataFrame({
    "user": ["alice", "bob", "bob", "dan", None],
    "email": ["ALICE@example.com", "bob@example.com", "bob+spam@example.com", None, "eva@example.com"],
    "age": [25, None, 27, 40, 22],
    "city": ["Dubai", "Dubai", "Dubai", "Sharjah", "Abu Dhabi"]
})


# ---------- Q1: Create & Inspect ----------
# Return the shape (rows, cols) and the dtypes of the 'customers' DataFrame as a Series.
def q1_shape_and_dtypes(df: pd.DataFrame):
    # TODO: return a tuple: (shape_tuple, dtypes_series)
    # shape_tuple example: (5, 4)
    # dtypes_series: df.dtypes
    return (df.shape, df.dtypes)


# ---------- Q2: Selection with loc/iloc ----------
# Using 'customers', return the name of the 3rd row (0-based) and all rows where city == "Dubai" with only ['name','city'].
def q2_indexing(df: pd.DataFrame):
    # TODO: return a tuple: (third_name, dubai_subset_df)
    return (customers.iloc[2]["name"], customers[customers.loc[:, "city"] == "Dubai"].loc[:, ["name", "city"]])


# ---------- Q3: Boolean filtering + query ----------
# From 'orders', select rows with status == "paid" and amount > 100 using boolean indexing (not query).
def q3_boolean_filter(df: pd.DataFrame):
    # TODO: return a filtered DataFrame
    return orders[orders["status"] == "paid"][orders["amount"] > 100]


# ---------- Q4: Handling missing data ----------
# In 'orders', fill missing 'amount' with the median amount (ignoring NaNs). Return the filled Series (order preserved).
def q4_fillna_median(df: pd.DataFrame):
    # TODO: compute median and fillna on 'amount'; return the resulting Series
    return orders["amount"].fillna(value=orders["amount"].median())


# ---------- Q5: GroupBy & Aggregation ----------
# For 'orders' (after filling NaNs with 0 for amount), compute total 'amount' per 'customer_id' and return a Series sorted desc.
def q5_groupby_sum(df: pd.DataFrame):
    # TODO: fillna(0) on 'amount', groupby customer_id and sum; sort descending by value
    return (df["amount"].fillna(0)
              .groupby(df["customer_id"]).sum()
              .sort_values(ascending=False))


# ---------- Q6: Merge / Join ----------
# Join 'customers' with aggregated order totals (from Q5) on 'customer_id' to get each customer's total_spent.
# Return a DataFrame with ['customer_id','name','total_spent'] sorted by total_spent desc, NaN -> 0.
def q6_join_customers_orders(customers_df: pd.DataFrame, orders_df: pd.DataFrame):
    # TODO: compute totals then merge; fill NaN totals with 0; sort desc
    totals = (orders_df["amount"].fillna(0)
                .groupby(orders_df["customer_id"]).sum()
                .rename("total_spent"))
    out = (customers_df.merge(totals, how="left", left_on="customer_id", right_index=True)
            .assign(total_spent=lambda d: d["total_spent"].fillna(0))
            .loc[:, ["customer_id", "name", "total_spent"]]
            .sort_values("total_spent", ascending=False))
    return out


# ---------- Q7: Pivot / Reshape ----------
# Build a pivot table of total quantity per customer_id and category using line_items + products.
# Return a DataFrame with index=customer_id, columns=category, values=total qty (sum), NaNs as 0.
def q7_pivot_qty_by_category(line_items_df: pd.DataFrame, orders_df: pd.DataFrame, products_df: pd.DataFrame):
    # Hints: merge line_items->orders (to get customer_id) and line_items->products (to get category), then pivot_table
    # TODO: return the pivoted DataFrame with fill_value=0
    li_o = line_items_df.merge(orders_df[["order_id", "customer_id"]], on="order_id", how="left")
    li_op = li_o.merge(products_df[["sku", "category"]], on="sku", how="left")
    return (li_op.pivot_table(index="customer_id", columns="category", values="qty",
                              aggfunc="sum", fill_value=0))


# ---------- Q8: Datetime operations ----------
# For 'orders', return a Series indexed by 'created_at'. The values should be daily total 'amount' (NaNs treated as 0).
# The index must be normalized to date (no time), sorted ascending.
def q8_daily_amount(df: pd.DataFrame):
    # TODO: set index to created_at.dt.normalize(), groupby index, sum filled amount
    tmp = df.copy()
    tmp["amount"] = tmp["amount"].fillna(0)
    dates = tmp["created_at"].dt.normalize()
    return tmp.groupby(dates)["amount"].sum().sort_index()


# ---------- Q9: String methods ----------
# Clean 'messy': lower-case emails, drop rows where email is NaN, remove "+..." from local-part (before "@").
# Return the cleaned DataFrame with a new column 'domain' extracted from email (part after "@").
def q9_clean_emails(df: pd.DataFrame):
    # TODO: use .str methods, .dropna on email, and regex to strip "+tag"
    out = df.dropna(subset=["email"]).copy()
    out["email"] = out["email"].str.lower()
    out["email"] = out["email"].str.replace(r"\+[^@]+(?=@)", "", regex=True)
    out["domain"] = out["email"].str.split("@").str[1]
    return out


# ---------- Q10: Apply / Transform ----------
# Using 'orders', create a new Series of the z-score of 'amount' (NaNs ignored for mean/std). Return the Series (aligned to df).
def q10_zscore_amount(df: pd.DataFrame):
    # TODO: (amount - mean) / std for non-NaN; keep NaN where amount is NaN
    mean = df["amount"].mean(skipna=True)
    std = df["amount"].std(skipna=True, ddof=1)
    return (df["amount"] - mean) / std


# ------------------ CHECKS ------------------

def _run_checks():
    # Q1
    shape, dtypes = q1_shape_and_dtypes(customers)
    assert shape == (5, 4)
    assert "name" in dtypes.index and dtypes["signup"].name in ("datetime64[ns]", "datetime64[ns, UTC]", "datetime64[ns, tz]")

    # Q2
    third_name, dubai_df = q2_indexing(customers)
    assert third_name == "Cathy"
    assert list(dubai_df.columns) == ["name", "city"]
    assert set(dubai_df["city"].unique()) == {"Dubai"}
    assert len(dubai_df) == 3

    # Q3
    q3 = q3_boolean_filter(orders)
    assert set(q3["status"].unique()) == {"paid"}
    assert (q3["amount"] > 100).all()
    assert set(q3["order_id"]) == {100, 105, 106}

    # Q4
    q4 = q4_fillna_median(orders.copy())
    median_expected = float(pd.Series([120.0, 50.0, 75.0, 0.0, 200.0, 130.0]).median())
    assert np.isclose(q4.iloc[1], median_expected)

    # Q5
    q5 = q5_groupby_sum(orders.copy())
    # fillna(0): amounts become [120,0,50,75,0,200,130]; totals by customer: {1:120,2:50,3:275,5:130}
    assert list(q5.index) == [3, 5, 1, 2]
    assert list(q5.values) == [275.0, 130.0, 120.0, 50.0]

    # Q6
    q6 = q6_join_customers_orders(customers.copy(), orders.copy())
    assert list(q6.columns) == ["customer_id", "name", "total_spent"]
    assert q6.iloc[0]["name"] in ("Cathy", "Eva") or q6.iloc[0]["total_spent"] in (275.0, 130.0)  # top spender is customer 3 (Cathy)

    # Q7
    q7 = q7_pivot_qty_by_category(line_items, orders, products)
    assert 3 in q7.index and "supplement" in q7.columns and "equipment" in q7.columns
    # customer 3 ordered: A1 x(3+2)=5 (supplement), B2 x1 (equipment)
    assert int(q7.loc[3, "supplement"]) == 5
    assert int(q7.loc[3, "equipment"]) == 1

    # Q8
    q8 = q8_daily_amount(orders.copy())
    assert q8.index.is_monotonic_increasing
    # day 2025-03-05 has amounts: NaN + 50 -> 50
    exp = pd.Timestamp("2025-03-05")
    assert float(q8.loc[exp]) == 50.0

    # Q9
    q9 = q9_clean_emails(messy.copy())
    assert "domain" in q9.columns
    assert set(q9["email"]) == {"alice@example.com", "bob@example.com", "bob@example.com", "eva@example.com"}
    assert set(q9["domain"]) == {"example.com"}

    # Q10
    q10 = q10_zscore_amount(orders.copy())
    mean = orders["amount"].mean(skipna=True)
    std = orders["amount"].std(skipna=True, ddof=1)
    expected = (orders["amount"] - mean) / std
    assert q10.equals(expected)

    print("✅ All 10 questions passed! Nice work.")

if __name__ == "__main__":
    try:
        _run_checks()
    except NotImplementedError:
        print("Implement the TODOs and run again!")
