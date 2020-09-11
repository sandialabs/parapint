nosetests -v -a '!parallel' -a n_procs=all -a n_procs=1 ./parapint/ --with-coverage --cover-package=parapint  --cover-html --cover-erase
mpirun -np 2 nosetests -a parallel,n_procs=all -a parallel,n_procs=2 ./parapint/ --with-coverage --cover-package=parapint  --cover-html
mpirun -np 3 nosetests -a parallel,n_procs=all -a parallel,n_procs=3 ./parapint/ --with-coverage --cover-package=parapint  --cover-html
mpirun -np 4 nosetests -a parallel,n_procs=all -a parallel,n_procs=4 ./parapint/ --with-coverage --cover-package=parapint  --cover-html
