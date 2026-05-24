import romjax as romx

import os

# from romjax.utils import LoggerConfig

os.chdir("demo")

routine = romx.load("demo.yml")
routine.run()
