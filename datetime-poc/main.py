import numpy as np

print("=" * 40)
print("       NumPy Datetime Operations        ")
print("=" * 40, "\n")


# 1. Creating np.datetime64 objects (different units)

precisions = {
    "Year": "2025", "Month": "2025-06", "Day": "2025-06-15",
    "Hour": "2025-06-15T14", "Minute": "2025-06-15T14:30",
    "Second": "2025-06-15T14:30:45", "Millisecond": "2025-06-15T14:30:45.500",
}
for label, val in precisions.items():
    print(f"{label}: {np.datetime64(val)}")

print(f"\nForced to seconds: {np.datetime64('2025-06-15', 's')}")
print(f"Forced to hours:   {np.datetime64('2025-06-15', 'h')}")

nat = np.datetime64("NaT")
print(f"\nNaT value: {nat}, is NaT: {np.isnat(nat)}\n")


# 2. Creating np.timedelta64 objects

timedeltas = [(5, "D"), (12, "h"), (90, "m"), (3600, "s"), (2, "W"), (3, "M"), (1, "Y")]
for val, unit in timedeltas:
    print(f"timedelta64({val}, '{unit}'): {np.timedelta64(val, unit)}")

td_one_day = np.timedelta64(1, "D")
for unit, label in [("h", "hours"), ("m", "minutes"), ("s", "seconds")]:
    print(f"1 day in {label}: {td_one_day / np.timedelta64(1, unit)}")
print()


# 3. Date arithmetic

base = np.datetime64("2025-06-15")
print(f"Base date: {base}")
print(f"+ 10 days:   {base + np.timedelta64(10, 'D')}")
print(f"- 30 days:   {base - np.timedelta64(30, 'D')}")
print(f"+ 4 weeks:   {base + np.timedelta64(4, 'W')}")
print(f"2025-06 + 3M: {np.datetime64('2025-06') + np.timedelta64(3, 'M')}")
print(f"2025 + 2Y:    {np.datetime64('2025') + np.timedelta64(2, 'Y')}")

date_a, date_b = np.datetime64("2025-12-31"), np.datetime64("2025-01-01")
diff = date_a - date_b
print(f"\n{date_a} - {date_b} = {diff} ({diff / np.timedelta64(1, 'D'):.0f} days, {diff / np.timedelta64(1, 'W'):.1f} weeks)")

time_a = np.datetime64("2025-06-15T08:00")
print(f"\n{time_a} + 90m = {time_a + np.timedelta64(90, 'm')}")
print(f"{time_a} + 5h30m = {time_a + np.timedelta64(5, 'h') + np.timedelta64(30, 'm')}\n")


# 4. Arrays of dates (np.arange)

daily = np.arange("2025-06-01", "2025-06-08", dtype="datetime64[D]")
monthly = np.arange("2025-01", "2026-01", dtype="datetime64[M]")
hourly = np.arange("2025-06-15T09:00", "2025-06-15T17:00", dtype="datetime64[h]")
weekly = np.arange("2025-06-01", "2025-08-01", dtype="datetime64[W]")
yearly = np.arange("2020", "2030", dtype="datetime64[Y]")

for label, arr in [("Daily", daily), ("Monthly", monthly), ("Hourly", hourly), ("Weekly", weekly), ("Yearly", yearly)]:
    print(f"{label} ({len(arr)}): {arr}")
print()


# 5. Comparing dates

d1, d2, d3 = np.datetime64("2025-06-15"), np.datetime64("2025-12-25"), np.datetime64("2025-06-15")
print(f"d1={d1}, d2={d2}, d3={d3}")
print(f"d1 < d2: {d1 < d2}, d1 == d3: {d1 == d3}, d1 >= d2: {d1 >= d2}")

dates = np.array(["2025-03-15", "2025-06-20", "2025-09-10", "2025-12-01"], dtype="datetime64")
threshold = np.datetime64("2025-07-01")
print(f"\nBefore {threshold}: {dates[dates < threshold]}")
print(f"After {threshold}:  {dates[dates >= threshold]}")
print(f"Min: {np.min(dates)}, Max: {np.max(dates)}, Sorted: {np.sort(dates)}\n")


# 6. Parsing date strings

iso_dates = np.array(["2025-01-15", "2025-06-30", "2025-12-25"], dtype="datetime64[D]")
mixed = np.array(["2025-01-15", "2025-06-30T12:00", "2025-12-25T08:30:00"], dtype="datetime64[s]")
print(f"ISO dates: {iso_dates}")
print(f"Mixed precision (as seconds): {mixed}")

date_str = "2025-06-15"
for unit in ["D", "M", "Y"]:
    print(f"'{date_str}' as [{unit}]: {np.datetime64(date_str, unit)}")

date_val = np.datetime64("2025-06-15")
print(f"datetime64 to str: '{str(date_val)}' (type: {type(str(date_val)).__name__})\n")


# 7. Business day functions

test_dates = np.arange("2025-06-09", "2025-06-16", dtype="datetime64[D]")
weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
is_bd = np.is_busday(test_dates)

print("Business day check (Jun 9-15):")
for i, d in enumerate(test_dates):
    print(f"  {d} ({weekdays[i]}): {is_bd[i]}")
print(f"Business days only: {test_dates[is_bd]}\n")

# busday_count
print(f"Business days in 2025: {np.busday_count('2025-01-01', '2025-12-31')}")
for q, (s, e) in enumerate([("2025-01-01", "2025-04-01"), ("2025-04-01", "2025-07-01"),
                              ("2025-07-01", "2025-10-01"), ("2025-10-01", "2026-01-01")], 1):
    print(f"  Q{q}: {np.busday_count(s, e)}")

# busday_offset
date = np.datetime64("2025-06-14")  # Saturday
print(f"\nFrom {date} (Sat): forward={np.busday_offset(date, 0, roll='forward')}, "
      f"backward={np.busday_offset(date, 0, roll='backward')}")
for n in [1, 5, -1]:
    print(f"  {n:+d} business days: {np.busday_offset(date, n, roll='forward')}")

# Custom holidays and weekmask
holidays = np.array(["2025-01-01", "2025-07-04", "2025-12-25"], dtype="datetime64[D]")
print(f"\nWith holidays {holidays}:")
print(f"  2025-07-04 is busday: {np.is_busday('2025-07-04', holidays=holidays)}")
print(f"  Business days in 2025: {np.busday_count('2025-01-01', '2025-12-31', holidays=holidays)}")

weekmask = "1111110"
print(f"Mon-Sat weekmask: Sat is busday={np.is_busday('2025-06-14', weekmask=weekmask)}, "
      f"working days in 2025={np.busday_count('2025-01-01', '2025-12-31', weekmask=weekmask)}\n")


# 8. Practical example: project timeline

print("--- Project Timeline ---\n")
project_start = np.datetime64("2025-06-02")
project_end = np.datetime64("2025-08-29")
company_holidays = np.array(["2025-07-04", "2025-07-05"], dtype="datetime64[D]")

calendar_days = int((project_end - project_start) / np.timedelta64(1, "D"))
total_bd = np.busday_count(project_start, project_end)
effective_bd = np.busday_count(project_start, project_end, holidays=company_holidays)
print(f"Start: {project_start}, End: {project_end}")
print(f"Calendar days: {calendar_days}, Business days: {total_bd}, Effective (with holidays): {effective_bd}\n")

# Milestones
print("Milestones:")
for bd in [10, 25, 40, 55, total_bd]:
    print(f"  Day {bd:3d}: {np.busday_offset(project_start, bd, holidays=company_holidays)}")

# Next business day resolution
print("\nNext business day resolution:")
for d in ["2025-06-14", "2025-06-15", "2025-07-04", "2025-06-18"]:
    d = np.datetime64(d)
    is_bd = np.is_busday(d, holidays=company_holidays)
    nxt = d if is_bd else np.busday_offset(d, 0, roll="forward", holidays=company_holidays)
    print(f"  {d} -> {nxt} ({'already busday' if is_bd else 'moved forward'})")

# Monthly breakdown
print("\nMonthly business days:")
for month in np.arange("2025-06", "2025-09", dtype="datetime64[M]"):
    ms = max(month.astype("datetime64[D]"), project_start)
    me = min((month + np.timedelta64(1, "M")).astype("datetime64[D]"), project_end)
    print(f"  {month}: {np.busday_count(ms, me, holidays=company_holidays)}")
print()
