"""Resume and checkpoint operations for optimization."""
import json
import os


def load_resume_checkpoint(resume_path, population, hyper_keys, full_bounds, incremental_enabled):
    """
    Load population state from resume checkpoint.
    
    Args:
        resume_path: Path to resume JSON file
        population: DEAP population to load into
        hyper_keys: Current list of active parameter names
        full_bounds: Dict of all parameter bounds
        incremental_enabled: Whether incremental optimization is active
    
    Returns:
        (start_gen: int, loaded_count: int, loaded_indices: set, actual_genome_size: int)
    """
    loaded_indices = set()
    start_gen = 0
    loaded_count = 0
    actual_genome_size = len(hyper_keys)
    
    try:
        print(f"\n[RESUME] Found resume file at: {resume_path}")
        print(f"[RESUME] Attempting to load population state...")
        
        with open(resume_path, 'r') as f:
            resume_data = json.load(f)
        
        saved_pop = resume_data.get("population", [])
        start_gen = resume_data.get("generation", -1) + 1
        
        if not saved_pop:
            print(f"[RESUME] WARN: Resume file found but contained no population data.")
            return start_gen, 0, loaded_indices, actual_genome_size
        
        # Detect genome size from saved population
        saved_genome_size = len(saved_pop[0]) if saved_pop else 0
        actual_genome_size = saved_genome_size
        
        print(f"[RESUME] Saved genome size: {saved_genome_size}, Current config expects: {len(hyper_keys)}")
        
        # Load individuals
        invalid_count = 0
        expanded_count = 0
        
        for i in range(min(len(population), len(saved_pop))):
            saved_ind = saved_pop[i]
            
            # Handle genome size mismatch
            if len(saved_ind) < len(hyper_keys):
                # Config has MORE parameters than saved
                new_param_count = len(hyper_keys) - len(saved_ind)
                print(f"[RESUME] Individual {i}: Config has {new_param_count} new parameters")
                
                if incremental_enabled:
                    # Don't expand - will use incremental addition
                    print(f"[RESUME] Incremental mode: Will add new parameters progressively")
                else:
                    # Expand immediately at midpoint
                    print(f"[RESUME] Non-incremental mode: Adding new parameters at midpoint")
                    new_params = hyper_keys[len(saved_ind):]
                    for param_name in new_params:
                        low, up = full_bounds[param_name]
                        midpoint = (low + up) / 2.0
                        if isinstance(low, int) and isinstance(up, int):
                            midpoint = int(round(midpoint))
                        saved_ind.append(midpoint)
                    expanded_count += 1
            
            elif len(saved_ind) > len(hyper_keys):
                # Config has FEWER parameters - truncate
                print(f"[RESUME] WARN: Individual {i} truncated from {len(saved_ind)} to {len(hyper_keys)} genes")
                saved_ind = saved_ind[:len(hyper_keys)]
            
            # Validate bounds (only for genes we're loading)
            valid = True
            genes_to_load = min(len(saved_ind), len(hyper_keys))
            
            for k in range(genes_to_load):
                val = saved_ind[k]
                param_name = hyper_keys[k] if k < len(hyper_keys) else f"param_{k}"
                if param_name in full_bounds:
                    low, up = full_bounds[param_name]
                    if not (low <= val <= up):
                        print(f"[RESUME] WARN: Individual {i} parameter '{param_name}' value {val} out of bounds [{low}, {up}]")
                        valid = False
                        invalid_count += 1
                        break
            
            # Load into population
            if valid:
                for k in range(genes_to_load):
                    population[i][k] = saved_ind[k]
                loaded_count += 1
                loaded_indices.add(i)
        
        print(f"[RESUME] SUCCESS: Loaded {loaded_count} individuals")
        if expanded_count > 0:
            print(f"[RESUME] Expanded {expanded_count} individuals with new parameters")
        if invalid_count > 0:
            print(f"[RESUME] Skipped {invalid_count} invalid individuals")
        print(f"[RESUME] Resuming from generation {start_gen}")
        
    except Exception as e:
        print(f"[RESUME] ERROR: Failed to load resume file: {e}")
    
    return start_gen, loaded_count, loaded_indices, actual_genome_size


def load_champion_parameters(params_path, hyper_keys, full_bounds):
    """
    Load champion parameters from file.
    
    Args:
        params_path: Path to champion parameters JSON
        hyper_keys: List of parameter names to load
        full_bounds: Dict of parameter bounds
    
    Returns:
        List of parameter values or None if failed
    """
    try:
        print(f"\n[CHAMPION LOAD] Found champion parameters file at: {params_path}")
        
        with open(params_path, 'r') as f:
            champ_dict = json.load(f)
        
        champ_ind = []
        for key in hyper_keys:
            if key not in champ_dict:
                print(f"[CHAMPION LOAD] WARN: Champion missing key '{key}'")
                return None
            
            val = champ_dict[key]
            
            # Handle special encodings
            if key == "use_log1p_features":
                val = 1 if val == ["typical_price"] else 0
            elif key == "positional_encoding":
                val = 1 if val else 0
            
            champ_ind.append(val)
        
        print(f"[CHAMPION LOAD] SUCCESS: Loaded champion with {len(champ_ind)} parameters")
        return champ_ind
        
    except Exception as e:
        print(f"[CHAMPION LOAD] ERROR: Failed to load champion: {e}")
        return None


def save_resume_checkpoint(resume_path, generation, population):
    """Save current population state to resume file."""
    try:
        resume_payload = {
            "generation": generation,
            "population": [list(ind) for ind in population]
        }
        
        # Atomic save
        temp_path = resume_path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(resume_payload, f, indent=2)
        os.replace(temp_path, resume_path)
        
        print(f"  Resume state saved to {resume_path}")
    except Exception as e:
        print(f"  Failed to save resume state: {e}")
