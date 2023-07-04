#include "mgr.hpp"

#include <madrona/macros.hpp>

#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weverything"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic pop
#endif

namespace nb = nanobind;

namespace GPUHideSeek {

NB_MODULE(gpu_hideseek_python, m) {
    nb::module_::import_("madrona_python");

    nb::class_<Manager> (m, "HideAndSeekSimulator")
        .def("__init__", [](Manager *self,
                            madrona::py::PyExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t render_width,
                            int64_t render_height, 
                            bool auto_reset,
                            bool enable_batch_render,
                            bool debug_compile) {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .renderWidth = (uint32_t)render_width,
                .renderHeight = (uint32_t)render_height,
                .autoReset = auto_reset,
                .enableBatchRender = enable_batch_render,
                .debugCompile = debug_compile,
            });
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("render_width"),
           nb::arg("render_height"),
           nb::arg("auto_reset") = false,
           nb::arg("enable_batch_render") = false,
           nb::arg("debug_compile") = false)
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("prep_counter_tensor", &Manager::prepCounterTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("agent_type_tensor", &Manager::agentTypeTensor)
        .def("agent_mask_tensor", &Manager::agentMaskTensor)
        .def("agent_data_tensor", &Manager::agentDataTensor)
        .def("box_data_tensor", &Manager::boxDataTensor)
        .def("ramp_data_tensor", &Manager::rampDataTensor)
        .def("visible_agents_mask_tensor", &Manager::visibleAgentsMaskTensor)
        .def("visible_boxes_mask_tensor", &Manager::visibleBoxesMaskTensor)
        .def("visible_ramps_mask_tensor", &Manager::visibleRampsMaskTensor)
        .def("global_positions_tensor", &Manager::globalPositionsTensor)
        .def("depth_tensor", &Manager::depthTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("lidar_tensor", &Manager::lidarTensor)
        .def("seed_tensor", &Manager::seedTensor)
    ;
}

}
