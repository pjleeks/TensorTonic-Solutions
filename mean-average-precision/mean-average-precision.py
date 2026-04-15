import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.
    
    Returns:
        tuple: (mAP_value, list_of_APs)
    """
    ap_per_query = []

    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        
        # 1. Sort labels by score in descending order
        sort_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sort_indices]
        
        # 2. Apply cutoff k if specified
        if k is not None:
            y_true_sorted = y_true_sorted[:k]
            
        # 3. Identify relevant items and their positions
        # Note: Total relevant items (R) is based on the full y_true, 
        # but the calculation only averages over those found within k.
        num_relevant_total = np.sum(y_true)
        
        if num_relevant_total == 0:
            ap_per_query.append(0.0)
            continue
            
        # Get positions of relevant items (1-based indexing for precision)
        relevant_indices = np.where(y_true_sorted == 1)[0]
        if len(relevant_indices) == 0:
            ap_per_query.append(0.0)
            continue
            
        # 4. Compute Precision at each relevant rank
        # p_at_k = (number of relevant items up to rank i) / (rank i)
        ranks = relevant_indices + 1
        precisions = np.arange(1, len(relevant_indices) + 1) / ranks
        
        # 5. Calculate AP for this query
        ap = np.sum(precisions) / num_relevant_total
        ap_per_query.append(ap)

    # Calculate mean across all queries
    mAP_value = np.mean(ap_per_query) if ap_per_query else 0.0
    
    return mAP_value, ap_per_query