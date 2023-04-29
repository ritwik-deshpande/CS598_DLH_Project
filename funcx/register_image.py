from funcx import FuncXClient


fx = FuncXClient()
uuid = fx.register_container("/projects/bbmi/bengal1/cs598_dlh_ritwikd2.sif",
                             container_type="singularity")
print("Container ID is ", uuid)

# 30fb631f-0f1c-4733-aebb-f267bb2e52a6  Ben
# d40eaddf-6b6b-410a-983b-622e6e5af32d  Sindhu
# ac70db7c-406d-4697-be1e-55cbf56b86f8  Ritwik