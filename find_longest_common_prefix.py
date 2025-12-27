#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to find the longest common prefix in the 'prompt' column of a CSV file.
"""

import argparse
import pandas as pd
import sys
import os


def find_longest_common_prefix(strings):
    """
    Find the longest common prefix among a list of strings.
    
    Args:
        strings: List of strings to find common prefix for
        
    Returns:
        The longest common prefix string
    """
    if not strings:
        return ""
    
    if len(strings) == 1:
        return strings[0]
    
    # Start with the first string as the candidate prefix
    prefix = strings[0]
    
    # Compare with each subsequent string
    for string in strings[1:]:
        # Find the common prefix between current prefix and this string
        i = 0
        min_len = min(len(prefix), len(string))
        while i < min_len and prefix[i] == string[i]:
            i += 1
        prefix = prefix[:i]
        
        # If no common prefix found, return empty string
        if not prefix:
            return ""
    
    return prefix


def main():
    parser = argparse.ArgumentParser(
        description='Find the longest common prefix in the "prompt" column of a CSV file.'
    )
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to the CSV file'
    )
    parser.add_argument(
        '--column',
        type=str,
        default='prompt',
        help='Column name to analyze (default: prompt)'
    )
    parser.add_argument(
        '--show-prefix',
        action='store_true',
        help='Display the full prefix (may be very long)'
    )
    parser.add_argument(
        '--save-prefix',
        type=str,
        default=None,
        help='Save the prefix to a text file'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    # Read CSV file
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Check if column exists
    if args.column not in df.columns:
        print(f"Error: Column '{args.column}' not found in CSV file.", file=sys.stderr)
        print(f"Available columns: {', '.join(df.columns)}", file=sys.stderr)
        sys.exit(1)
    
    # Get all prompt values (remove NaN values)
    prompts = df[args.column].dropna().astype(str).tolist()
    
    if not prompts:
        print(f"Error: No valid values found in column '{args.column}'.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Analyzing {len(prompts)} rows from column '{args.column}'...")
    print(f"Finding longest common prefix...")
    
    # Find longest common prefix
    common_prefix = find_longest_common_prefix(prompts)

    df['ratio'] = df.apply(lambda x: len(common_prefix)/len(x['prompt']),axis=1)
    avg_ratio = df['ratio'].mean()
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Longest common prefix length: {len(common_prefix)} characters")
    print(f"Total rows analyzed: {len(prompts)}")
    print(f"Average ratio of common prefix length to prompt length: {avg_ratio}")

    if args.show_prefix:
        print(f"\nLongest common prefix:")
        print("-"*80)
        print(common_prefix)
        print("-"*80)
    else:
        # Show first 200 characters as preview
        preview = common_prefix[:200] if len(common_prefix) > 200 else common_prefix
        print(f"\nPrefix preview (first 200 chars):")
        print("-"*80)
        print(preview)
        if len(common_prefix) > 200:
            print(f"... ({len(common_prefix) - 200} more characters)")
        print("-"*80)
        print("\nUse --show-prefix to display the full prefix.")
    
    # Save prefix to file if requested
    if args.save_prefix:
        try:
            with open(args.save_prefix, 'w', encoding='utf-8') as f:
                f.write(common_prefix)
            print(f"\nPrefix saved to: {args.save_prefix}")
        except Exception as e:
            print(f"Error saving prefix to file: {e}", file=sys.stderr)
    
    print("="*80)


if __name__ == "__main__":
    main()

