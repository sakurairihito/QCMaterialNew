## outlines 
The program can be run without MPI:

```
julia --project=@. dimer_plus_minus.jl plus_true minus_false sp_tau.txt >> output
```

This is the brief explanation of the command.
--project=@.: sets the project root to the current directory.
dimer_plus_minus.jl: the name of the Julia script to execute.
plus_true, minus_false, and sp_tau.txt: the arguments passed to the script.

"plus_true minus_false" means that the program will perform imaginary-time evolution from 0 to half of beta.
On the other hand, 
"plus_false minus_true" means that the program will perform imaginary-time evolution from 0 to -(half of beta).

## details 
In "plus_false minus_true", you should use the functions "write_to_txt_half_to_beta" to convert the G(tau) date into textfile.
This function change the range of imaginary time, half of beta to beta from 0 to -(half of beta).