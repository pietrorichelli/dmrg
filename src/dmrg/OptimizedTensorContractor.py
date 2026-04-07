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
    
    pop_idx = {
        True: lambda i, j: (j, i),
        False: lambda i, j: (i, j)
    }

    def __init__(self, optimize='optimal'):
        # Initialize the contraction strategy and cached paths.
        self.optimize = optimize
        self._cache = {}  # Cache: (equation, rounded_shapes) -> (contraction_plan, transpose_axes)

    @staticmethod
    def _round_dim_to_10(dim):
        # Round large dimensions to the nearest multiple of ten.
        if dim < 10:
            return dim
        return int(round(dim / 10.0) * 10)

    def _rounded_shapes_key(self, tensors):
        # Build a cache key from tensor shapes rounded by dimension.
        return tuple(
            tuple(self._round_dim_to_10(dim) for dim in tensor.shape)
            for tensor in tensors
        )
    
    def contract(self, equation, *tensors):
        # Execute the contraction using a cached or newly computed plan.
        cache_key = (equation, self._rounded_shapes_key(tensors))

        try:
            contraction_plan, transpose_axes = self._cache[cache_key]
        except KeyError:
            self._cache[cache_key] = self._precompute_plan(equation, cache_key[1])
            contraction_plan, transpose_axes = self._cache[cache_key]
        
        temp_tensors = list(tensors)
        result = None
        
        for (i, j), (axes_i, axes_j) in contraction_plan:
            t_i = temp_tensors[i]
            t_j = temp_tensors[j]
            result = np.tensordot(t_i, t_j, axes=(axes_i, axes_j))
            
            first, second = self.pop_idx[i < j](i, j)
            temp_tensors.pop(first)
            temp_tensors.pop(second)
            temp_tensors.append(result)
        
        return np.transpose(result, transpose_axes)
    
    def _precompute_plan(self, equation, rounded_shapes):
        # Precompute the tensordot sequence and final transpose order.
        
        def einsum_to_tensordot(eq, path):
            # Translate an einsum contraction path into tensordot steps.
            inputs, output = eq.split("->")
            output_inds = list(output)
            tensors_list = [list(x) for x in inputs.split(",")]
            plan = []
            
            for path_step in path:
                i, j = path_step
                inds_i = tensors_list[i]
                inds_j = tensors_list[j]
                shared = [x for x in inds_i if x in inds_j]
                
                axes_i = tuple(inds_i.index(x) for x in shared)
                axes_j = tuple(inds_j.index(x) for x in shared)
                
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
                
                first, second = self.pop_idx[i < j](i, j)
                tensors_list.pop(first)
                tensors_list.pop(second)
                tensors_list.append(new_inds)
            
            final_inds = tensors_list[0]
            transpose_axes = tuple(final_inds.index(x) for x in output_inds)
            
            return plan, transpose_axes
        
        dummy_tensors = [np.empty(shape) for shape in rounded_shapes]
        path, _ = oe.contract_path(equation, *dummy_tensors, optimize=self.optimize)
        contraction_plan, transpose_axes = einsum_to_tensordot(equation, path)
        
        return contraction_plan, transpose_axes

_default_contractors = {}

def contract(equation, *tensors, optimize='optimal'):
    # Reuse one default contractor per optimization strategy.
    contractor = _default_contractors.get(optimize)
    if contractor is None:
        contractor = OptimizedTensorContractor(optimize=optimize)
        _default_contractors[optimize] = contractor
    return contractor.contract(equation, *tensors)