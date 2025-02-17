#pragma once

#include "BiasField.hpp"
#include "DarkField.hpp"
#include "FlatField.hpp"
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

// 工作流配置
struct WorkflowConfig {
    BiasConfig bias_config;
    FlatConfig flat_config;
    DefectPixelMapper::Config dark_config;
    
    bool enable_gpu = true;          // 启用GPU加速
    int thread_count = 4;            // 并行线程数
    size_t memory_limit = 8192;      // 内存限制(MB)
    bool auto_optimize = true;       // 自动优化参数
    std::string cache_dir = "cache"; // 缓存目录
    
    std::string to_string() const;
};

// 处理状态
struct ProcessStatus {
    enum class Stage {
        IDLE,
        BIAS,
        DARK,
        FLAT,
        LIGHT,
        COMPLETE,
        ERROR
    };
    
    Stage current_stage = Stage::IDLE;
    float progress = 0.0f;
    std::string message;
    std::string error;
};

// 校准结果
struct CalibrationResult {
    cv::Mat master_bias;
    cv::Mat master_dark;
    cv::Mat master_flat;
    QualityMetrics bias_quality;
    FlatQualityMetrics flat_quality;
    
    void save(const std::string& dir);
    void load(const std::string& dir);
};

class AstroProcessor {
public:
    explicit AstroProcessor(const WorkflowConfig& config);
    
    // 异步处理接口
    void start_processing(
        const std::vector<std::string>& bias_files,
        const std::vector<std::string>& dark_files,
        const std::vector<std::string>& flat_files,
        const std::vector<std::string>& light_files,
        const std::string& output_dir
    );
    
    // 状态查询
    ProcessStatus get_status() const;
    
    // 控制接口
    void pause();
    void resume();
    void cancel();
    
    // 结果获取
    CalibrationResult get_calibration();
    
    // 优化建议
    std::string get_optimization_suggestions();

private:
    WorkflowConfig config_;
    ProcessStatus status_;
    std::unique_ptr<std::thread> worker_;
    std::atomic<bool> paused_{false};
    std::atomic<bool> cancelled_{false};
    CalibrationResult result_;
    
    // 内部处理方法
    void process_workflow(
        const std::vector<std::string>& bias_files,
        const std::vector<std::string>& dark_files,
        const std::vector<std::string>& flat_files,
        const std::vector<std::string>& light_files,
        const std::string& output_dir
    );
    
    // 资源管理
    void manage_memory();
    void setup_gpu();
    void clear_cache();
    
    // 优化方法
    void optimize_parameters();
    void analyze_system_resources();
    void schedule_tasks();
    
    // 进度更新
    void update_progress(float progress, const std::string& message);
    void report_error(const std::string& error);
};
