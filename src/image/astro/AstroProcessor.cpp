#include "AstroProcessor.hpp"
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <spdlog/spdlog.h>
#include <thread>

namespace fs = std::filesystem;

// WorkflowConfig实现
std::string WorkflowConfig::to_string() const {
    return fmt::format(
        "工作流配置:\n"
        "GPU加速: {}\n"
        "线程数: {}\n"
        "内存限制: {} MB\n"
        "自动优化: {}\n"
        "缓存目录: {}",
        enable_gpu, thread_count, memory_limit,
        auto_optimize, cache_dir
    );
}

// CalibrationResult实现
void CalibrationResult::save(const std::string& dir) {
    fs::create_directories(dir);
    
    cv::imwrite(fs::path(dir) / "master_bias.tiff", master_bias);
    cv::imwrite(fs::path(dir) / "master_dark.tiff", master_dark);
    cv::imwrite(fs::path(dir) / "master_flat.tiff", master_flat);
    
    // 保存质量指标
    std::ofstream quality_file(fs::path(dir) / "quality_metrics.txt");
    quality_file << "偏置帧质量指标:\n" << bias_quality.to_string() << "\n\n"
                << "平场质量指标:\n" << flat_quality.to_string();
}

void CalibrationResult::load(const std::string& dir) {
    master_bias = cv::imread(fs::path(dir) / "master_bias.tiff", cv::IMREAD_UNCHANGED);
    master_dark = cv::imread(fs::path(dir) / "master_dark.tiff", cv::IMREAD_UNCHANGED);
    master_flat = cv::imread(fs::path(dir) / "master_flat.tiff", cv::IMREAD_UNCHANGED);
    
    // TODO: 加载质量指标
}

// AstroProcessor实现
AstroProcessor::AstroProcessor(const WorkflowConfig& config)
    : config_(config) {
    spdlog::info("初始化天文图像处理器");
    
    if (config_.auto_optimize) {
        analyze_system_resources();
        optimize_parameters();
    }
    
    if (config_.enable_gpu) {
        setup_gpu();
    }
}

void AstroProcessor::start_processing(
    const std::vector<std::string>& bias_files,
    const std::vector<std::string>& dark_files,
    const std::vector<std::string>& flat_files,
    const std::vector<std::string>& light_files,
    const std::string& output_dir
) {
    if (worker_ && worker_->joinable()) {
        spdlog::warn("已有处理任务在运行");
        return;
    }
    
    cancelled_ = false;
    paused_ = false;
    
    worker_ = std::make_unique<std::thread>(&AstroProcessor::process_workflow,
        this, bias_files, dark_files, flat_files, light_files, output_dir);
}

ProcessStatus AstroProcessor::get_status() const {
    return status_;
}

void AstroProcessor::pause() {
    paused_ = true;
}

void AstroProcessor::resume() {
    paused_ = false;
}

void AstroProcessor::cancel() {
    cancelled_ = true;
}

CalibrationResult AstroProcessor::get_calibration() {
    return result_;
}

std::string AstroProcessor::get_optimization_suggestions() {
    std::vector<std::string> suggestions;
    
    // 分析内存使用
    size_t available_memory = analyze_system_resources();
    if (available_memory < config_.memory_limit) {
        suggestions.push_back(
            fmt::format("建议降低内存限制至 {} MB", available_memory)
        );
    }
    
    // 分析GPU能力
    if (config_.enable_gpu && !cv::cuda::getCudaEnabledDeviceCount()) {
        suggestions.push_back("系统未检测到GPU，建议禁用GPU加速");
    }
    
    // 分析CPU核心数
    int cpu_cores = std::thread::hardware_concurrency();
    if (config_.thread_count > cpu_cores) {
        suggestions.push_back(
            fmt::format("建议将线程数从 {} 调整为 {}", 
                config_.thread_count, cpu_cores)
        );
    }
    
    return fmt::format("{}", fmt::join(suggestions, "\n"));
}

void AstroProcessor::process_workflow(
    const std::vector<std::string>& bias_files,
    const std::vector<std::string>& dark_files,
    const std::vector<std::string>& flat_files,
    const std::vector<std::string>& light_files,
    const std::string& output_dir
) {
    try {
        // 创建输出目录
        fs::create_directories(output_dir);
        
        // 处理偏置帧
        status_.current_stage = ProcessStatus::Stage::BIAS;
        update_progress(0.0f, "处理偏置帧...");
        
        BiasProcessor bias_processor(config_.bias_config);
        std::vector<cv::Mat> bias_frames;
        for (const auto& file : bias_files) {
            bias_frames.push_back(cv::imread(file, cv::IMREAD_UNCHANGED));
        }
        result_.master_bias = bias_processor.create_master_bias(bias_frames);
        result_.bias_quality = bias_processor.analyze_noise(
            result_.master_bias, bias_frames);
            
        // 处理暗场
        if (cancelled_) return;
        status_.current_stage = ProcessStatus::Stage::DARK;
        update_progress(0.25f, "处理暗场...");
        
        DefectPixelMapper dark_processor(config_.dark_config);
        std::vector<cv::Mat> dark_frames;
        for (const auto& file : dark_files) {
            dark_frames.push_back(cv::imread(file, cv::IMREAD_UNCHANGED));
        }
        dark_processor.build_defect_map(dark_frames);
        result_.master_dark = dark_processor.correct_image(dark_frames[0]);
        
        // 处理平场
        if (cancelled_) return;
        status_.current_stage = ProcessStatus::Stage::FLAT;
        update_progress(0.5f, "处理平场...");
        
        FlatFieldProcessor flat_processor(config_.flat_config);
        std::vector<cv::Mat> flat_frames;
        for (const auto& file : flat_files) {
            flat_frames.push_back(cv::imread(file, cv::IMREAD_UNCHANGED));
        }
        result_.master_flat = flat_processor.process(
            flat_frames, result_.master_bias, result_.master_dark);
        result_.flat_quality = flat_processor.getQualityMetrics();
        
        // 处理光场
        if (cancelled_) return;
        status_.current_stage = ProcessStatus::Stage::LIGHT;
        update_progress(0.75f, "校正光场...");
        
        size_t processed = 0;
        #pragma omp parallel for if(config_.thread_count > 1)
        for (size_t i = 0; i < light_files.size(); ++i) {
            if (cancelled_) continue;
            
            cv::Mat light = cv::imread(light_files[i], cv::IMREAD_UNCHANGED);
            light = apply_flat_correction(light, result_.master_flat,
                result_.master_bias, result_.master_dark);
                
            std::string output_name = fs::path(light_files[i]).stem().string() 
                + "_calibrated.tiff";
            cv::imwrite(fs::path(output_dir) / output_name, light);
            
            #pragma omp atomic
            ++processed;
            
            update_progress(0.75f + 0.25f * processed / light_files.size(),
                fmt::format("已处理 {}/{} 张光场", processed, light_files.size()));
        }
        
        // 保存校准结果
        if (!cancelled_) {
            result_.save(fs::path(output_dir) / "calibration");
            status_.current_stage = ProcessStatus::Stage::COMPLETE;
            update_progress(1.0f, "处理完成");
        }
        
    } catch (const std::exception& e) {
        report_error(e.what());
    }
    
    // 清理缓存
    clear_cache();
}

void AstroProcessor::manage_memory() {
    // TODO: 实现内存管理
}

void AstroProcessor::setup_gpu() {
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        cv::cuda::setDevice(0);
        spdlog::info("GPU加速已启用");
    } else {
        config_.enable_gpu = false;
        spdlog::warn("未检测到GPU，已禁用GPU加速");
    }
}

void AstroProcessor::clear_cache() {
    try {
        fs::remove_all(config_.cache_dir);
    } catch (const std::exception& e) {
        spdlog::warn("清理缓存失败: {}", e.what());
    }
}

void AstroProcessor::optimize_parameters() {
    // TODO: 实现参数优化
}

void AstroProcessor::analyze_system_resources() {
    // TODO: 实现系统资源分析
}

void AstroProcessor::schedule_tasks() {
    // TODO: 实现任务调度
}

void AstroProcessor::update_progress(float progress, const std::string& message) {
    status_.progress = progress;
    status_.message = message;
    spdlog::info("{}: {:.1f}%", message, progress * 100);
}

void AstroProcessor::report_error(const std::string& error) {
    status_.current_stage = ProcessStatus::Stage::ERROR;
    status_.error = error;
    spdlog::error("处理错误: {}", error);
}
