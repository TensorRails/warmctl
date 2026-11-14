A MI350X pod's view of the KFD topology (an unprivileged container that sees
one of eight GPUs), reconstructed from an environment probe: node 3 is the
pod's GPU; nodes 2 and 4-9 are the other GPUs, whose properties the container
may list but not read (EPERM, simulated by the test); nodes 0 and 1 are CPU
nodes. The gpu_id and unique_id are anonymised; device_id (0x75a0), vendor_id
and the memory bank were not in the probe and are typical values. Replace this
fixture with a full capture of /sys/class/kfd/kfd/topology from an MI350X.
