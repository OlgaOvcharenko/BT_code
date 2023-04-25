# BS_OlgaOvcharenko
To run example with tax dataset simply call 
`python scale_modify.py`

`python 3.8,
pandas 1.4.1,
numpy 1.20.1`

# Scale one dataset -> scale_modify.py
To scale call scale_modify.run_default with clean_dataset path, dity_dataset path and scaling factor. 
All other arguments are optional (clean_header: int = 0, dirty_header: int = 0, clean_sep: str = ',', 
dirty_sep: str = ',', fds_file_path: str = None). 
New dataset will be generated together with its statistics and error distribution.
Mean, distincts, error distribution are preserved.

# Components
## Error distribution computation
error_distribution.py
## Scale clean dataset
scale_dataset.py 
3 methods to scale dataset: replicate int times and 
* random choose rows (default) 
* hash (computationally hard) 
* slice exact part of clean
## Generate errors 
error_sequence_generator.py
Generate outliers, MV, typos, replacements, swaps
## Validate 
compare_error_dist.py

