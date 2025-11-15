import pandas as pd
from pybaseball import batting_stats

def find_team_leaders():
    """
    Prompts user for team, year, and min PA, then prints the
    leader in various stats from that team's qualified players.
    """
    
    # --- 1. Get User Input ---
    print("--- Baseball Team Stat Leader Finder ---")
    try:
        # Use .upper() to make team abbreviation case-insensitive
        team_abbr = input("Enter Team Abbreviation (e.g., BOS, NYY, LAD): ").upper()
        year = int(input("Enter Year (e.g., 2023): "))
        min_pa = int(input("Enter Minimum Plate Appearances (e.g., 400): "))
    except ValueError:
        print("\nError: Year and PA must be whole numbers.")
        return

    # --- 2. Fetch Data ---
    print(f"\nFetching data for {year}...")
    try:
        # Fetch all player data for the specified season
        # This returns a pandas DataFrame
        all_data = batting_stats(year, year)
        
        if all_data is None or all_data.empty:
            print("No data returned from pybaseball. Check the year or library.")
            return

    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Please check your internet connection and the pybaseball library.")
        return
        
    # --- 3. Filter Data ---
    
    # Filter all data down to just the specified team
    # We use .str.upper() to handle any potential case issues in the data source
    team_data = all_data[all_data['Team'].str.upper() == team_abbr]

    if team_data.empty:
        print(f"Error: No data found for team '{team_abbr}' in {year}.")
        print("Please check the team abbreviation.")
        return

    # Filter that team data for players who meet the minimum PA
    qualified_players = team_data[team_data['PA'] >= min_pa]

    # --- 4. Check if any players qualified ---
    if qualified_players.empty:
        print(f"\nNo players found for {team_abbr} in {year} with at least {min_pa} PA.")
        print("Try lowering the PA requirement.")
        return

    # --- 5. Display Qualified Players ---
    print(f"\nFound {len(qualified_players)} qualified player(s) for {team_abbr} ({year}, {min_pa}+ PA):")
    
    # We use to_string(index=False) to print a clean list of names and their PAs
    print(qualified_players[['Name', 'PA']].to_string(index=False))

    # --- 6. Find and Print Leaders ---
    print("\n--- Category Leaders ---")

    # Define the stats we want to check.
    # Added 'Barrel%' and 'HardHit%' to the list.
    stats_to_check = [
        'HR', 'RBI', 'R', 'SB', 
        'AVG', 'OBP', 'SLG', 'OPS', 
        'Barrel%', 'HardHit%', 
        'wOBA', 'xwOBA', 'wRC+', 'WAR'
    ]
    
    # Create a cleaner DataFrame with just the players and stats we need
    leaderboard_data = qualified_players[['Name'] + stats_to_check]

    for stat in stats_to_check:
        if stat not in leaderboard_data.columns:
            print(f"Note: '{stat}' not available in this dataset. Skipping.")
            continue
            
        # Sort by the current stat, highest first
        sorted_players = leaderboard_data.sort_values(by=stat, ascending=False)
        
        # Get the leader (the first row after sorting)
        leader = sorted_players.iloc[0]
        
        player_name = leader['Name']
        stat_value = leader[stat]
        
        # Print the leader's name and their value for that stat
        # We add some formatting to make it look nice
        if stat in ['AVG', 'OBP', 'SLG', 'OPS', 'wOBA', 'xwOBA']:
            # Format ratios to 3 decimal places
            print(f"{stat}: {player_name} ({stat_value:.3f})")
        elif stat in ['Barrel%', 'HardHit%']:
            # Format percentages to 1 decimal place with a % sign
            print(f"{stat}: {player_name} ({stat_value:.1f}%)")
        else:
            # Format counting stats and WAR/wRC+
            print(f"{stat}: {player_name} ({stat_value})")

    print("\nDone.")

# --- Run the main function when the script is executed ---
if __name__ == "__main__":
    find_team_leaders()