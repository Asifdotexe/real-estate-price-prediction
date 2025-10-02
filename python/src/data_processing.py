import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def convert_sqft(value: str) -> float | None:
    """Convert various area representations to square feet

    :param value: Area string, possibly a range or with units.
    :returns: Converted area in square feet, or None if unconvertible.
    """
    # Non-string or empty inputs can't be processed meaningfully.
    if not isinstance(value, str) or not value.strip():
        return None

    value = value.strip()

    # If the value is a range like "2100 - 2850",
    # we approximate by averaging the two numbers
    if "-" in value:
        tokens = value.strip("-")
        if len(tokens) == 2:
            first = float(tokens[0].strip())
            second = float(tokens[1].strip())
            return (first + second) / 2

    # Extract the first numeric part to interpret the value, even if it includes a unit.
    numeric_match = re.match(r"([\d\.]+)", value)
    if not numeric_match:
        # If there is no number, we cannot proceed with conversions.
        return None

    try:
        # Defensive check in case float conversion fails
        numeric_value = float(numeric_match.group(1))
    except ValueError:
        return None

    value_lower = value.lower()

    # Define unit to sqft conversion mapping
    unit_to_sqft = {
        "sq. meter": 10.7639,
        "sq meter": 10.7639,
        "sqm": 10.7639,
        "sq. yard": 9,
        "sq yards": 9,
        "sq yard": 9,
        # fallback for less specific yard mentions
        "yard": 9,
        "acre": 43560,
        "ground": 2400,
        "guntha": 1089,
        "cent": 435.6,
        # fallback for less specific meter mentions
        "meter": 10.7639
    }

    # Checking if any known unit is in the string and convert accordingly
    for unit, factor in unit_to_sqft.items():
        if unit in value_lower:
            return numeric_value * factor

    # Handling "perch" specifically (damn this was confusing).
    # Skip conversion as it's unclear
    if "perch" in value_lower:
        return None

    # If no unit matched, assume it is already in sqft.
    return numeric_value

def remove_bhk_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outlier rows where a higher BHK apartment is priced less per square foot
    than the average price per square foot of a lower BHK apartment in the same location

    Why is this needed?
        In most reasonable markets, a 3 BHK flat should not be cheaper (per sqft)
        than a 2 BHK in the same locality, unless it's an anomaly.
        This rule helps filter out such inconsistencies that could hurt model accuracy.

    :param df: Dataframe with "location", "bhk", "price_per_sqft" columns
    :returns: DataFrame with outlier row removed
    """
    indices_to_exclude = np.array([], dtype=int)

    for location in df["location"].unique():
        location_df = df[df["location"] == location]
        bhk_price_stats: dict[int, dict[str, float]] = {}

        # calculate the mean and standard deviation of price per sqft
        # for each BHK level in this location.
        for bhk_level in location_df["bhk"].unique():
            bhk_df = location_df[location_df["bhk"] == bhk_level]
            bhk_price_stats[bhk_level] = {
                "mean": bhk_df["price_per_sqft"].mean(),
                "std": bhk_df["price_per_sqft"].std()
            }

        # if a higher BHK flat is priced lower than the average of the next lower BHK,
        # it's probably an outlier unless it's backed by sufficient data.
        for bhk_level in location_df["bhk"].unique():
            lower_bhk_level = bhk_level - 1
            if lower_bhk_level in bhk_price_stats:
                lower_bhk_mean = bhk_price_stats[lower_bhk_level]["mean"]
                current_bhk_df = location_df[location_df["bhk"] == bhk_level]

                if not np.isnan(lower_bhk_level) and len(current_bhk_df) > 5:
                    outlier_indices = current_bhk_df[
                        current_bhk_df["price_per_sqft"] < lower_bhk_mean
                    ].index.values

                    indices_to_exclude = np.concatenate((indices_to_exclude,
                                                         outlier_indices))
    return df.drop(indices_to_exclude.astype(int), axis="index")

def clean_and_cluster_locations(df: pd.DataFrame,
                                location_col: str = "location") -> pd.DataFrame:
    """
    Clean and cluster location data in a DataFrame.

    This function standardizes the formatting of location names, removes duplicate entries,
    and assigns each location to a geographic zone (East, West, South, North, Central) using exact matching.
    Locations that do not match any predefined zone are categorized under "Other".

    :param df: The input DataFrame containing location data.
    :param location_col: The name of the column in `df` that contains the location names.
    :returns: A cleaned and enriched DataFrame with an additional "zone" column.
    """

    # Why: It's important to standardize the formatting to ensure consistency across names
    # (e.g., "whitefield", " WhiteField ", and "Whitefield" should be treated the same).
    df[location_col] = (
        df[location_col]
        .astype(str)  # Ensure all entries are strings
        .str.strip()  # Remove leading/trailing spaces
        .str.replace(r"\s+", " ",
                     regex=True)  # Collapse multiple spaces into one
        .str.title()
    # Capitalize appropriately (e.g., "jp nagar" -> "Jp Nagar")
    )

    # Why: Duplicate entries waste space and can bias analysis, so we keep only the first unique one.
    df = df.drop_duplicates(subset=[location_col]).reset_index(drop=True)

    # Why: We want to enrich the dataset by grouping each location into a broader zone.
    # This helps with regional analysis and insights without manually checking each locality.
    predefined_zone_map = {
        "East": ["Whitefield", "Kr Puram", "Marathahalli", "Kadugodi", "Itpl",
                 "Brookefield", "Hoodi", "Ramamurthy Nagar", "Mahadevpura"],
        "West": ["Rajaji Nagar", "Vijayanagar", "Magadi Road", "Malleswaram",
                 "Nagarbhavi", "Basaveshwaranagar", "Chandra Layout"],
        "South": ["Jp Nagar", "Jayanagar", "Banashankari", "Kanakapura Road",
                  "Bannerghatta Road", "Btm Layout", "Electronic City",
                  "Arekere", "Hulimavu"],
        "North": ["Yelahanka", "Hebbal", "Jakkur", "Sahakara Nagar",
                  "Nagavara", "Rt Nagar", "Kodigehalli"],
        "Central": ["Mg Road", "Ulsoor", "Richmond Town", "Frazer Town",
                    "Shivajinagar", "Indiranagar", "Koramangala"]
    }

    # Why: By flattening the zone dictionary, we create a simple lookup for efficient mapping.
    # This avoids having to loop over nested lists or write complex matching logic.
    area_to_zone_map = {
        locality: zone
        for zone, localities in predefined_zone_map.items()
        for locality in localities
    }

    # Why: Assigning zones directly by map ensures clean logic, and fallback to "Other" captures everything else.
    df["zone"] = df[location_col].map(area_to_zone_map).fillna("Other")

    return df