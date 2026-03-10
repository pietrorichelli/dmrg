# SPDX-FileCopyrightText: 2025-present pietrorichelli <richelli.pietro@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Simple logging utility for writing output to both terminal and file."""


class Logger:
    """Dual-output logger that writes to both terminal and file."""
    
    def __init__(self, filepath):
        """
        Initialize logger with a file path.
        
        Parameters:
        -----------
        filepath : str
            Path to the log file to write to.
        """
        self.filepath = filepath
        self.file = open(filepath, 'w')
    
    def print(self, *args, **kwargs):
        """
        Print to both terminal (stdout) and log file.
        
        Parameters:
        -----------
        *args : objects
            Arguments to print (same as built-in print)
        **kwargs : dict
            Keyword arguments (end, sep, etc.) same as built-in print
            Note: flush parameter is respected, but file flushing only happens 
            on newlines or when explicitly requested for efficiency
        """
        # Extract flush parameter (default False for file I/O efficiency)
        flush_requested = kwargs.pop('flush', False)
        
        # Print to terminal with requested flush
        print(*args, **kwargs, flush=flush_requested)
        
        # Print to file
        output = ' '.join(str(arg) for arg in args)
        end_char = kwargs.get('end', '\n')
        self.file.write(output + end_char)
        
        # Only flush on newlines or when explicitly requested
        # This significantly improves performance by batching disk writes
        if flush_requested or end_char == '\n':
            self.file.flush()
    
    def close(self):
        """Close the log file."""
        self.file.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
