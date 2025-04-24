import torch
import time
import functools  # Used for functools.wraps


def monitor_runtime_memory(fit_func):
    """
    Decorator to monitor runtime and peak GPU memory usage for a PyTorch function.

    Assumes the decorated function's first two arguments are 'model' and 'device'.
    """

    @functools.wraps(fit_func)  # Preserves original function metadata (name, docstring)
    def wrapper(model, device, *args, **kwargs):
        """
        Wrapper function that performs monitoring.
        """
        print(f"\n--- Preparing Monitoring for {fit_func.__name__} ---")

        # --- 1. Measure Runtime & Memory ---
        start_time = time.time()
        peak_memory_gb = "N/A"  # Default value if not using CUDA

        # Memory Measurement Setup (Peak GPU Memory)
        if device.type == "cuda":
            # Reset peak memory stats for the target device before the function call
            torch.cuda.reset_peak_memory_stats(device)
            memory_before_gb = torch.cuda.memory_allocated(device) / (1024**3)
            print(
                f"GPU Memory allocated before '{fit_func.__name__}': {memory_before_gb:.4f} GB"
            )
        else:
            print(
                "Device is CPU. Peak GPU memory usage cannot be measured with torch.cuda."
            )
            print(
                "Consider using 'tracemalloc' for peak RAM usage monitoring (requires separate implementation)."
            )

        # --- Execute the Original Function ---
        print(f"\n--- Running {fit_func.__name__} ---")
        # Execute the decorated function (the actual 'fit' logic)
        result = fit_func(model, device, *args, **kwargs)
        print(f"--- Finished {fit_func.__name__} ---")

        # --- Finalize Measurements & Report ---
        end_time = time.time()
        runtime_seconds = end_time - start_time

        # Memory Measurement Finalization (Peak GPU Memory)
        if device.type == "cuda":
            # Get peak memory allocated during the function call [[3]]
            peak_memory_bytes = torch.cuda.max_memory_allocated(device)
            peak_memory_gb = peak_memory_bytes / (1024**3)
            memory_after_gb = torch.cuda.memory_allocated(device) / (1024**3)
            print(
                f"GPU Memory allocated after '{fit_func.__name__}': {memory_after_gb:.4f} GB"
            )

        # --- Print Results ---
        print("\n--- Monitoring Results ---")
        print(f"Function Executed: {fit_func.__name__}")
        print(f"Total Runtime: {runtime_seconds:.2f} seconds")

        if device.type == "cuda":
            print(
                f"Peak GPU Memory Allocated during execution: {peak_memory_gb:.4f} GB"
            )
        else:
            print(f"Peak GPU Memory Allocated: N/A (CPU execution)")
        print("-" * 25)

        # Return the original function's result, if any
        return result

    # The decorator returns the wrapper function
    return wrapper
