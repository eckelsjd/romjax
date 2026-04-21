import subprocess

argv = ['codex']
prompt_file = 'tasks.md'
with open(prompt_file, "r", encoding="utf-8") as fd:
    process = subprocess.Popen(argv, stdin=fd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    # subprocess.run(['codex'], text=True, check=False, stderr=subprocess.STDOUT)
    process.wait()