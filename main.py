# This wrapper automatically loads the API

import sys, select, time, os, subprocess


p = 1
if p!=1:
  print("You must set an API_SECRET using the Secrets tool", file=sys.stderr)

else:

  print("Booting into API Server…")
  time.sleep(1)
  os.system('clear')
  print("BOT API SERVER RUNNING")
  p = subprocess.Popen([sys.executable, 'server.py'],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT,
                       text=True)
  while True:
    line = p.stdout.readline()
    if not line: break
    print(line.strip())
