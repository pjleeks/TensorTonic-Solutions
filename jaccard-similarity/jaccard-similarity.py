def jaccard_similarity(list_a, list_b):
    # Convert lists to sets to handle duplicates and allow set operations
    set_a = set(list_a)
    set_b = set(list_b)
    
    # Calculate the size of the intersection (A ∩ B)
    intersection = len(set_a.intersection(set_b))
    
    # Calculate the size of the union (A ∪ B)
    union = len(set_a.union(set_b))
    
    # Edge case: If both sets are empty, the union will be 0
    if union == 0:
        return 0.0
        
    # Apply the formula: J(A, B) = |A ∩ B| / |A ∪ B|
    return float(intersection) / union

# Example Usage:
user_1_items = ["laptop", "mouse", "keyboard", "mouse"]
user_2_items = ["mouse", "monitor", "keyboard"]

similarity = jaccard_similarity(user_1_items, user_2_items)
print(f"Jaccard Similarity: {similarity:.4f}")