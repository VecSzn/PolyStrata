#include "test_utils.hpp"
#include "strata/gpu/metal.hpp"
#include <cassert>
#include <thread>
#include <atomic>
int main(int argc,char**) {
    if(argc>1) {
        g::Stream s;ck(g::stream_create(&s));Buffer x(128);g::Event event;ck(g::event_create(&event));
        ck(g::memset_async(x.ptr,0,128,s));g::metal::testing::fail_next_completion();ck(g::event_record(event,s));
        if(g::event_synchronize(event)!=g::kErrorLaunchFailure)return 1;
        if(g::device_synchronize()!=g::kErrorLaunchFailure)return 2;
        if(g::get_last_error()!=g::kErrorLaunchFailure)return 3;
        g::event_destroy(event);g::stream_destroy(s);std::puts("failed completion wakes waiters: OK");return 0;
    }
    g::Stream a,b;ck(g::stream_create(&a));ck(g::stream_create(&b));
    Buffer x(128),y(128);g::Event e;ck(g::event_create(&e));
    ck(g::memset_async(x.ptr,7,128,a));ck(g::event_record(e,a));
    ck(g::stream_wait_event(b,e));ck(g::memcpy_async(y.ptr,x.ptr,128,g::kDeviceToDevice,b));
    ck(g::device_synchronize());for(auto v:y.read<unsigned char>(128))assert(v==7);
    auto before=g::metal::stats();ck(g::memset_async(x.ptr,3,128,a));ck(g::metal::submit(a));
    auto after=g::metal::stats();assert(after.host_syncs==before.host_syncs);
    ck(g::stream_synchronize(a));
    g::Graph graph;g::GraphExec exec;
    ck(g::begin_capture(a));ck(g::memcpy_async(y.ptr,x.ptr,128,g::kDeviceToDevice,a));
    ck(g::metal::submit(a));ck(g::event_record(e,a));ck(g::end_capture(a,&graph));
    ck(g::graph_instantiate(&exec,graph));ck(g::graph_destroy(graph));
    ck(g::free_device(x.ptr));x.ptr=nullptr; // graph owns its captured source
    ck(g::event_destroy(e)); // captured event owns its state
    ck(g::graph_launch(exec,b));ck(g::device_synchronize());
    for(auto v:y.read<unsigned char>(128))assert(v==3);
    ck(g::graph_exec_destroy(exec));
    g::Event survivor;ck(g::event_create(&survivor));ck(g::event_record(survivor,a));
    ck(g::stream_destroy(a));ck(g::event_synchronize(survivor));ck(g::event_destroy(survivor));
    // A free must flush a different explicit stream, not just the legacy stream.
    void* z;ck(g::malloc_device_raw(&z,128));ck(g::memset_async(z,8,128,b));ck(g::free_device(z));
    g::metal::set_scoped_residency(true);
    ck(g::memset_async(y.ptr,9,128,b));ck(g::stream_synchronize(b));
    for(auto v:y.read<unsigned char>(128))assert(v==9);
    // Scoped copies do not keep unrelated buffers alive through a graph.
    const auto base=g::metal::testing::allocated_bytes();
    void* unrelated;ck(g::malloc_device_raw(&unrelated,2<<20));
    ck(g::begin_capture(b));ck(g::memset_async(y.ptr,10,128,b));ck(g::end_capture(b,&graph));
    ck(g::graph_instantiate(&exec,graph));ck(g::graph_destroy(graph));
    ck(g::free_device(unrelated));assert(g::metal::testing::allocated_bytes()==base);
    ck(g::graph_launch(exec,b));ck(g::stream_synchronize(b));
    assert(g::metal::testing::last_residency_bytes()==128);ck(g::graph_exec_destroy(exec));
    // Unknown kernels retain their fallback resources past every Submit boundary,
    // including destruction of the graph before the last command buffer completes.
    std::string error;
    auto* pipeline=g::metal::pipeline("lifetime",R"(
        #include <metal_stdlib>
        using namespace metal;
        struct A { device int* p; };
        kernel void increment(constant A& a [[buffer(0)]]) { a.p[0]+=1; }
    )","increment",error);
    if(!pipeline)throw std::runtime_error(error);
    void* value;ck(g::malloc_device_raw(&value,128));ck(g::memset_device(value,0,128));
    struct Args { void* value; } args{value};
    ck(g::begin_capture(b));
    ck(g::metal::dispatch(b,pipeline,&args,sizeof(args),{1,1,1},{1,1,1}));ck(g::metal::submit(b));
    ck(g::metal::dispatch(b,pipeline,&args,sizeof(args),{1,1,1},{1,1,1}));
    ck(g::end_capture(b,&graph));ck(g::graph_instantiate(&exec,graph));ck(g::graph_destroy(graph));
    ck(g::free_device(value));ck(g::graph_launch(exec,b));ck(g::graph_exec_destroy(exec));
    ck(g::memcpy_async(y.ptr,value,4,g::kDeviceToDevice,b));ck(g::stream_synchronize(b));
    assert(y.read<int>(1)[0]==2);
    // Pointer tables explicitly declare indirect dependencies, including updates
    // after graph capture. Unrelated allocations stay outside scoped residency.
    Buffer table(sizeof(void*)),source(128),replacement(128);
    int number=41;source.put(&number,4);number=42;replacement.put(&number,4);
    const void* source_ptr=source.ptr;table.put(&source_ptr,sizeof(source_ptr));
    ck(g::metal::register_indirect(table.ptr,&source_ptr,1));
    pipeline=g::metal::pipeline("indirect_lifetime",R"(
        #include <metal_stdlib>
        using namespace metal;
        struct A { device void* table; device int* out; };
        kernel void read_table(constant A& a [[buffer(0)]]) { a.out[0]=static_cast<device int* const*>(a.table)[0][0]; }
    )","read_table",error);
    if(!pipeline)throw std::runtime_error(error);
    struct IndirectArgs {void* table;void* out;} indirect{table.ptr,y.ptr};
    const void* direct[]={table.ptr,y.ptr};
    ck(g::begin_capture(b));
    ck(g::metal::dispatch(b,pipeline,&indirect,sizeof(indirect),{1,1,1},{1,1,1},0,{direct,2,true,table.ptr}));
    ck(g::end_capture(b,&graph));ck(g::graph_instantiate(&exec,graph));ck(g::graph_destroy(graph));
    ck(g::graph_launch(exec,b));ck(g::stream_synchronize(b));assert(y.read<int>(1)[0]==41);
    source_ptr=replacement.ptr;table.put(&source_ptr,sizeof(source_ptr));
    ck(g::metal::register_indirect(table.ptr,&source_ptr,1));
    ck(g::graph_launch(exec,b));ck(g::stream_synchronize(b));assert(y.read<int>(1)[0]==42);
    ck(g::graph_exec_destroy(exec));
    // Explicit-stream encoding can run alongside device synchronization.
    std::atomic<bool> done=false;
    std::thread producer([&]{for(int i=0;i<50;++i)ck(g::memset_async(y.ptr,i,128,b));done=true;});
    while(!done.load())ck(g::device_synchronize());producer.join();ck(g::device_synchronize());
    for(auto v:y.read<unsigned char>(128))assert(v==49);
    ck(g::stream_destroy(b));
    std::puts("stream, event, graph and buffer lifetimes: OK");
}
