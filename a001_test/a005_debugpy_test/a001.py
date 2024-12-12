import debugpy
debugpy.listen(("localhost", 5678))
debugpy.wait_for_client()

print(123)
print(456)
pass
