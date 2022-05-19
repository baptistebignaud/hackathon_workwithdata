# hackathon_workwithdata

# Usefull commands:

Be carefull, the repository is built to give the possibility to use a GPU. If you don't have a GPU on your computer, please comment lines below _deploy_ in the docker-compose file or it won't work.

## Build the container<br>

> `docker-compose build ` <br>

## Up containers<br>

> `docker-compose up -d` <br>

## Stop containers<br>

> `docker-compose stop` <br>

## Open a shell inside the main container<br>

> `docker-compose exec datathon_workwithdata sh `

## Run jupyter lab from the container<br>

> `jupyter lab --ip 0.0.0.0 --allow-root`