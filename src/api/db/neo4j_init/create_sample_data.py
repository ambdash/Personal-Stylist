## STUB EXAMPLE OF PROCESSING DATA FOR NEO4J
import pandas as pd
import os

def create_sample_data():
    output_dir = "data/neo4j"
    os.makedirs(output_dir, exist_ok=True)
    
    items_data = {
        'item_id': range(1, 11),
        'name': ['Black Jeans', 'White T-shirt', 'Leather Jacket', 'Sneakers', 'Dress', 
                 'Blazer', 'Skirt', 'Boots', 'Sweater', 'Shirt'],
        'category': ['Pants', 'Top', 'Outerwear', 'Shoes', 'Dress', 
                    'Outerwear', 'Bottom', 'Shoes', 'Top', 'Top'],
        'style': ['Casual', 'Basic', 'Edgy', 'Sporty', 'Feminine', 
                 'Business', 'Feminine', 'Edgy', 'Casual', 'Business'],
        'color': ['Black', 'White', 'Black', 'White', 'Red', 
                 'Navy', 'Black', 'Black', 'Gray', 'White'],
        'season': ['All', 'All', 'Fall,Winter', 'All', 'Summer', 
                  'All', 'All', 'Fall,Winter', 'Fall,Winter', 'All']
    }
    pd.DataFrame(items_data).to_csv(f'{output_dir}/items.csv', index=False)
    
    # Create style_rules.csv
    style_rules_data = {
        'style': ['Casual', 'Business', 'Edgy', 'Feminine', 'Sporty'],
        'description': [
            'Relaxed and comfortable everyday wear',
            'Professional and polished look',
            'Bold and alternative style',
            'Soft and graceful appearance',
            'Athletic and dynamic look'
        ],
        'key_pieces': [
            'Jeans,T-shirt,Sneakers',
            'Blazer,Dress pants,Formal shoes',
            'Leather jacket,Boots,Dark colors',
            'Dresses,Skirts,Delicate accessories',
            'Athletic wear,Sneakers,Comfortable fabrics'
        ]
    }
    pd.DataFrame(style_rules_data).to_csv(f'{output_dir}/style_rules.csv', index=False)
    
    # Create compatibility.csv
    compatibility_data = {
        'item1_id': [1, 1, 2, 2, 3, 5],
        'item2_id': [2, 4, 3, 4, 4, 8],
        'compatibility_score': [0.9, 0.8, 0.85, 0.7, 0.95, 0.8]
    }
    pd.DataFrame(compatibility_data).to_csv(f'{output_dir}/compatibility.csv', index=False)

if __name__ == "__main__":
    create_sample_data() 