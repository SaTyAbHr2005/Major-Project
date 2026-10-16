"""
Flower in the application (Module 9): one Flower server per federation job whose protocol is
"flower". The server carries the model to the hospitals over gRPC and runs the planned rounds
automatically, while the existing Module 9 engine still opens rounds, validates every update,
aggregates, versions the global models and applies the round plan.
"""
