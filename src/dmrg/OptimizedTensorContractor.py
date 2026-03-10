import numpy as np
import opt_einsum as oe

class OptimizedTensorContractor:
    """
    Efficient tensor contraction using opt_einsum path optimization.
    
    Caches path computation per equation to avoid redundant work.
    For small tensor counts, uses shrinking list method (fast & simple).
    
    Parameters
    ----------
    optimize : str, optional
        Path optimization strategy. Options:
        - 'optimal': Find best path (guaranteed best FLOP count, fast for <~12 tensors)
        - 'greedy': Fast heuristic
        - 'dynamic-programming': Good balance between quality and speed
    
    Methods
    -------
    contract(equation, *tensors)
        Execute contraction with given equation and tensors, return result.
        Caches path computation for repeated calls with same equation.
    
    Examples
    --------
    >>> contractor = OptimizedTensorContractor(optimize='optimal')
    >>> result = contractor.contract('abc,dae->be', ten0, ten1)
    >>> result.shape  # (4, 25)
    """
    
    # Global attribute: maps condition (i < j) to lambda that returns pop order
    pop_idx = {
        True: lambda i, j: (j, i),
        False: lambda i, j: (i, j)
    }

    # Initialize caching storage for contraction plans
    def __init__(self, optimize='optimal'):
        self.optimize = optimize
        self._cache = {}  # Cache: equation -> (contraction_plan, transpose_axes)
    
    def contract(self, equation, *tensors):
        """
        Execute the optimized tensor contraction.
        
        Parameters
        ----------
        equation : str
            Einstein summation convention string (e.g., 'abc,dae->be')
        *tensors : np.ndarray
            Variable number of tensors to contract (must match equation)
        
        Returns
        -------
        np.ndarray
            Contracted result with indices matching the output specification
        """
        # Check cache for this equation
        try:
            contraction_plan, transpose_axes = self._cache[equation]
        except KeyError:
            self._cache[equation] = self._precompute_plan(equation)
            contraction_plan, transpose_axes = self._cache[equation]
        
        # Execute contraction using shrinking list (fast for small n)
        temp_tensors = list(tensors)
        result = None
        
        for step_idx, ((i, j), (axes_i, axes_j)) in enumerate(contraction_plan):
            t_i = temp_tensors[i]
            t_j = temp_tensors[j]
            result = np.tensordot(t_i, t_j, axes=(axes_i, axes_j))
            
            # Remove contracted tensors using global pop_idx (largest index first)
            first, second = self.pop_idx[i < j](i, j)
            temp_tensors.pop(first)
            temp_tensors.pop(second)
            
            temp_tensors.append(result)
        
        return np.transpose(result, transpose_axes)
    
    # Precompute contraction plan and transpose axes for efficient reuse across identical equations
    def _precompute_plan(self, equation):
        
        def einsum_to_tensordot(eq, path):
            inputs, output = eq.split("->")
            output_inds = list(output)
            tensors_list = [list(x) for x in inputs.split(",")]
            plan = []
            
            for path_step in path:
                i, j = path_step
                inds_i = tensors_list[i]
                inds_j = tensors_list[j]
                shared = [x for x in inds_i if x in inds_j]
                
                axes_i = tuple([inds_i.index(x) for x in shared])
                axes_j = tuple([inds_j.index(x) for x in shared])
                
                axis_pairs = list(zip(axes_i, axes_j))
                axis_pairs.sort(key=lambda x: x[0])
                if axis_pairs:
                    axes_i, axes_j = zip(*axis_pairs)
                    axes_i, axes_j = tuple(axes_i), tuple(axes_j)
                
                plan.append(((i, j), (axes_i, axes_j)))
                
                new_inds = (
                    [x for x in inds_i if x not in shared] +
                    [x for x in inds_j if x not in shared]
                )
                
                # Use global pop_idx dictionary
                first, second = self.pop_idx[i < j](i, j)
                tensors_list.pop(first)
                tensors_list.pop(second)
                
                tensors_list.append(new_inds)
            
            final_inds = tensors_list[0]
            transpose_axes = tuple([final_inds.index(x) for x in output_inds])
            
            return plan, transpose_axes
        
        # Get path from opt_einsum
        num_tensors = len(equation.split("->")[0].split(","))
        dummy_tensors = [np.empty((2,)*len(idx)) for idx in equation.split("->")[0].split(",")]
        path, _ = oe.contract_path(equation, *dummy_tensors, optimize=self.optimize)
        
        # Build contraction plan
        contraction_plan, transpose_axes = einsum_to_tensordot(equation, path)
        
        return contraction_plan, transpose_axes