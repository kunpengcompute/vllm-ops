#include "cpu_utils.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <omp.h>
#include <sched.h>
#include <unistd.h>

// 全局变量定义
std::vector<int> cpu_ids = {};
bool i8mm_flag = false;
int nrc_value = 2;  // 默认值为2

// 字符串分割函数
std::vector<std::string> split_str(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream token_stream(str);
    while (std::getline(token_stream, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

// 获取总线程数
int get_total_thread_num() {
    static bool initialized = false;
    static int cached_thread_num = 1;
    
    if (!initialized) {
        const char* env = getenv("OMP_NUM_THREADS");
        if (env && *env) {
            cached_thread_num = std::stoi(env);
            omp_set_num_threads(cached_thread_num);
        } else {
            cached_thread_num = omp_get_max_threads();
        }
        initialized = true;
    }
    
    return cached_thread_num;
}

// 解析序列字符串
std::vector<int> parse_sequence(const std::string& input) {
    std::vector<int> result;
    std::vector<std::string> parts = split_str(input, ',');
    for (std::vector<std::string>::iterator it = parts.begin(); it != parts.end(); ++it) {
        const std::string& part = *it;
        size_t colon_pos = part.find(":");
        if (colon_pos != std::string::npos) {
            std::string range = part.substr(0, colon_pos);
            std::string step_str = part.substr(colon_pos + 1);
            size_t dash_pos = range.find('-');
            if (dash_pos == std::string::npos) {
                continue;
            }
            int start = std::stoi(range.substr(0, dash_pos));
            int end = std::stoi(range.substr(dash_pos + 1));
            int step = std::stoi(step_str);

            for (int num = start; num <= end; num += step) {
                result.push_back(num);
            }
        }
        else if ((colon_pos = part.find('-')) != std::string::npos) {
            int start = std::stoi(part.substr(0, colon_pos));
            int end = std::stoi(part.substr(colon_pos + 1));

            for (int i = start; i <= end; ++i) {
                result.push_back(i);
            }
        }
        else {
            result.push_back(std::stoi(part));
        }
    }
    return result;
}

// 获取CPU亲和性配置
void get_affinity_cpus(std::vector<int>& cpu_ids) {
    static bool init = false;
    if (!init) {
        init = true;
        const char* env = getenv("CUSTOM_CPU_AFFINITY");
        if (env && *env) {
            cpu_ids = parse_sequence(env);
        }
        std::sort(cpu_ids.begin(), cpu_ids.end());
    }
}

// 初始化进程亲和性
void init_process_affinity() {
    if (cpu_ids.size() == 0) {
        return;
    }
    int total_thread_num = get_total_thread_num();
    if (total_thread_num % cpu_ids.size() != 0) {
        std::cout << "Error! OMP_NUM_THREADS: " << total_thread_num << ", ";
        std::cout << "CPU_IDs: ";
        for (int i = 0; i < cpu_ids.size(); ++i) {
            std::cout << cpu_ids[i] << " ";
        }
        exit(0);
    }
    int current_thread_id = omp_get_thread_num();
    int per_cpu = total_thread_num / cpu_ids.size();

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_ids[current_thread_id / per_cpu], &mask);

    int ret = sched_setaffinity(0, sizeof(cpu_set_t), &mask);
    if (ret == -1) {
        std::cout << "Error! Binding core failed!\n";
        exit(0);
    }
}

// 检测i8mm指令集支持
bool detect_i8mm_support() {
    FILE* pipe = popen("lscpu | grep i8mm", "r");
    if (pipe == nullptr) {
        std::cerr << "Warning: Failed to execute lscpu command, assuming i8mm not supported" << std::endl;
        return false;
    }
    
    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    
    int ret_code = pclose(pipe);
    
    // 如果命令执行成功且有输出，说明支持i8mm
    bool has_i8mm = (ret_code == 0 && !result.empty());
    
    std::cout << "i8mm detection: " << (has_i8mm ? "supported" : "not supported") << std::endl;
    
    return has_i8mm;
}

// 初始化i8mm标志
bool init_i8mm_flag() {
    static bool initialized = false;
    if (!initialized) {
        i8mm_flag = detect_i8mm_support();
        initialized = true;
    }
    return i8mm_flag;
}

// 获取NRC环境变量
int get_nrc_value() {
    static bool init = false;
    if (!init) {
        init = true;
        const char* env = getenv("NRC");
        if (env && *env) {
            int value = std::stoi(env);
            if (value == 2 || value == 4) {
                nrc_value = value;
            } else {
                std::cout << "Warning: Invalid NRC value '" << value << "', using default value 2" << std::endl;
                nrc_value = 2;
            }
        }
        std::cout << "NRC value: " << nrc_value << std::endl;
    }
    return nrc_value;
}

// 初始化工作分配结构体
void init_work_divider(WorkDivider *divider, int numas) {
    divider->num_numas = numas;
    divider->num_threads = omp_get_num_threads();
    if (divider->num_threads % divider->num_numas != 0) {
        fprintf(stderr, "nthreads (%d) %% numas (%d) != 0\n", divider->num_threads, divider->num_numas);
        exit(1);
    }
    divider->tid = omp_get_thread_num();
    // printf("tid:%d, num_threads:%d \n", divider->tid, divider->num_threads);
    divider->threads_per_numa = divider->num_threads / divider->num_numas;
    divider->my_numa = divider->tid / divider->threads_per_numa;
    divider->tid_in_numa = divider->tid % divider->threads_per_numa;
}

// 分配所有工作
void divide_all_work(const WorkDivider *divider, int total_workitems, SingleNumaWorkRange *pstSingleRange)
{
    int work_per_thread = total_workitems / divider->num_threads;
    int work_remaining = total_workitems % divider->num_threads;
    if (work_remaining == 0) {
        pstSingleRange->begin_thread = divider->tid * work_per_thread;
        pstSingleRange->end_thread = divider->tid * work_per_thread + work_per_thread;
        pstSingleRange->work_per_thread = work_per_thread;
    } else if (divider->tid < work_remaining) {
        pstSingleRange->begin_thread = divider->tid * work_per_thread + divider->tid;
        pstSingleRange->end_thread = (divider->tid + 1) * work_per_thread + (divider->tid + 1);
        pstSingleRange->work_per_thread = work_per_thread + 1;
    } else {
        pstSingleRange->begin_thread = divider->tid * work_per_thread + work_remaining;
        pstSingleRange->end_thread = (divider->tid + 1) * work_per_thread + work_remaining;
        pstSingleRange->work_per_thread = work_per_thread;
    }
    return;
}

// 分配单NUMA节点的工作
void divide_work_first_numa(const WorkDivider *divider, int total_workitems, SingleNumaWorkRange *pstSingleRange)
{
    if (divider->my_numa == 0) {
        int work_per_thread = total_workitems / divider->threads_per_numa;
        int work_remaining = total_workitems % divider->threads_per_numa;
        if (work_remaining == 0) {
            pstSingleRange->begin_thread = divider->tid * work_per_thread;
            pstSingleRange->end_thread = divider->tid * work_per_thread + work_per_thread;
            pstSingleRange->work_per_thread = work_per_thread;
        } else if (divider->tid < work_remaining) {
            pstSingleRange->begin_thread = divider->tid * work_per_thread + divider->tid;
            pstSingleRange->end_thread = (divider->tid + 1) * work_per_thread + (divider->tid + 1);
            pstSingleRange->work_per_thread = work_per_thread + 1;
        } else {
            pstSingleRange->begin_thread = divider->tid * work_per_thread + work_remaining;
            pstSingleRange->end_thread = (divider->tid + 1) * work_per_thread + work_remaining;
            pstSingleRange->work_per_thread = work_per_thread;
        }
        return;
    }

    pstSingleRange->begin_thread = 0;
    pstSingleRange->end_thread = 0;
    pstSingleRange->work_per_thread = 0;
}

// 分配所有NUMA节点的工作
void divide_work_all_numas(const WorkDivider *divider, int total_workitems, MultiNumaWorkRange *pstNulRange)
{
    int max_workitems_per_numa = (total_workitems - 1) / divider->num_numas + 1;
    int workitem_numa_begin = divider->my_numa * max_workitems_per_numa;
    int workitem_numa_end = workitem_numa_begin + max_workitems_per_numa;
    if (workitem_numa_end > total_workitems) {
        workitem_numa_end = total_workitems;
    }
    int workitems_my_numa = workitem_numa_end - workitem_numa_begin;
    int max_workitems_per_thread = (workitems_my_numa - 1) / divider->threads_per_numa + 1;
    int begin = divider->tid_in_numa * max_workitems_per_thread;
    int end = begin + max_workitems_per_thread;
    if (end > workitems_my_numa) {
        end = workitems_my_numa;
    }

    pstNulRange->begin_numa = workitem_numa_begin;
    pstNulRange->end_numa = workitem_numa_end;
    pstNulRange->work_per_numa = max_workitems_per_numa;
    pstNulRange->begin_thread = begin;
    pstNulRange->end_thread = end;
    pstNulRange->work_per_thread = end - begin;
} 