
import psycopg2
import randomname
import useful_stuff

from useful_stuff import logger

# a lot of this code is written explicitly over multiple lines to be easier to read (e.g. name creation)

# connect to the database
config = useful_stuff.load_confg()
conn = psycopg2.connect(**config)
logger ('connected to database')

# open a cursor to perform operatoins
cur = conn.cursor()

# notes no how to use random names: https://pypi.org/project/randomname/
name = randomname.get_name()
name = name.replace("-", " ") # because name generator puts a hyphen between two words
name = name.title() # capitalize each letter
email = name.lower().replace(" ", "@") + ".com" # convert the name into an email of sorts
logger("name {}, email {}".format(name,email));

# insert a person into the people table
# https://www.psycopg.org/docs/usage.html#the-problem-with-the-query-parameters says to pass params as a second argument
#insertQuery = "insert into andrew.people (name, email) values ('{}', '{}')".format(name,email)
#logger("insertQuery -> " + insertQuery)
insertQuery = "insert into andrew.people (name, email) values (%s, %s)"
data = (name, email)
cur.execute(insertQuery, data)

# query and print out the people in the table
selectQuery = "select * from andrew.people"
cur.execute(selectQuery)
# can use the cursor itself as the iterator, because it is iterable (https://www.psycopg.org/docs/cursor.html#cursor-iterable)
for record in cur:
    print(record)

cur.close()
conn.commit() # alternately, could set conn.autocommit = True above
conn.close()
logger ('closed database connection')