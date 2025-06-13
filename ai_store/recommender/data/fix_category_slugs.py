# fix_category_slugs.py
import csv

INPUT_CSV = './books.csv'
OUTPUT_CSV = './books_fixed.csv'

CATEGORY_MAP = {
    '1': 'fiction',
    '2': 'non-fiction',
    '3': 'business',
    '4': 'technology',
    '5': 'history',
    '6': 'science',
    '7': 'children',
    '8': 'fantasy',
}

def fix_category_slugs():
    with open(INPUT_CSV, newline='', encoding='utf-8') as infile, \
         open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            # Skip empty rows (causing ValueError)
            if None in row:
                continue

            old_value = row['category_slug'].strip()
            new_slug = CATEGORY_MAP.get(old_value, old_value)

            row['category_slug'] = new_slug
            writer.writerow(row)

    print(f"✅ Fixed categories written to: {OUTPUT_CSV}")

if __name__ == '__main__':
    fix_category_slugs()
