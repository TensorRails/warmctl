A MI350X pod's view of the KFD topology (an unprivileged container that sees
one of eight GPUs), with the values a pod read on 2026-11-15: node 3 is the
pod's GPU (device_id 0x75a0, whose render node names it "AMD Instinct MI350
OAM"); nodes 2 and 4-9 are the other GPUs, whose properties the container may
list but not read (EPERM, simulated by the test); nodes 0 and 1 are CPU nodes.
The gpu_id and unique_id are anonymised; properties warmctl does not read are
left out.
The render minors of the seven hidden nodes are synthetic for the host-wide test.
