CC = icc
CFLAGS = -std=c++11 -Ofast
LIBS = -ltbb -lpapi -lpfm

INCLUDE_PAPI = /opt/apps/papi-5.5.1/src 
INCLUDE_TBB = /opt/intel/compilers_and_libraries_2018.0.128/linux/compiler/include/icc
LIBRARIES_PFM4 = /opt/apps/papi-5.5.1/src/libpfm4/lib 
LIBRARIES_PAPI = /opt/apps/papi-5.5.1/src
OBJECTS = born_profiler.cc

all: vector vector_threads vector_tiling_threads vector_512

vector: vector.cc 
	$(CC) $(CFLAGS) -I$(INCLUDE_PAPI) -I$(INCLUDE_TBB) -L$(LIBRARIES_PFM4) -L$(LIBRARIES_PAPI) $(LIBS) -o vector vector.cc $(OBJECTS)

vector_threads: vector_threads.cc 
	$(CC) $(CFLAGS) -I$(INCLUDE_PAPI) -I$(INCLUDE_TBB) -L$(LIBRARIES_PFM4) -L$(LIBRARIES_PAPI) $(LIBS) -o vector_threads vector_threads.cc $(OBJECTS)

vector_tiling_threads: vector_tiling_threads.cc
	$(CC) $(CFLAGS) -I$(INCLUDE_PAPI) -I$(INCLUDE_TBB) -L$(LIBRARIES_PFM4) -L$(LIBRARIES_PAPI) $(LIBS) -o vector_tiling_threads vector_tiling_threads.cc $(OBJECTS)

vector512_threads: vector512_threads.cc 
	$(CC) $(CFLAGS) -I$(INCLUDE_PAPI) -I$(INCLUDE_TBB) -L$(LIBRARIES_PFM4) -L$(LIBRARIES_PAPI) $(LIBS) -o vector512_threads vector512_threads.cc $(OBJECTS)

vector512_tiling_threads: vector512_tiling_threads.cc
	$(CC) $(CFLAGS) -I$(INCLUDE_PAPI) -I$(INCLUDE_TBB) -L$(LIBRARIES_PFM4) -L$(LIBRARIES_PAPI) $(LIBS) -o vector512_tiling_threads vector512_tiling_threads.cc $(OBJECTS)

vector_512: vector_512.cc 
	$(CC) $(CFLAGS) -I$(INCLUDE_PAPI) -I$(INCLUDE_TBB) -L$(LIBRARIES_PFM4) -L$(LIBRARIES_PAPI) $(LIBS) -o vector512 vector512.cc $(OBJECTS)

copy_blocked: copy_blocked.cc 
	$(CC) $(CFLAGS) -I$(INCLUDE_TBB) -ltbb -o copy_blocked copy_blocked.cc

copy_blocked_prop: copy_blocked_prop.cc 
	$(CC) $(CFLAGS) -I$(INCLUDE_TBB) -ltbb -o copy_blocked_prop copy_blocked_prop.cc

copy3D: copy3D.cc 
	$(CC) $(CFLAGS) -I$(INCLUDE_TBB) -ltbb -o copy3D copy3D.cc

naive_prop: naive_prop.cc 
	$(CC) $(CFLAGS) -I$(INCLUDE_PAPI) -I$(INCLUDE_TBB) -L$(LIBRARIES_PFM4) -L$(LIBRARIES_PAPI) $(LIBS) -o naive_prop naive_prop.cc $(OBJECTS)

