#pragma once

#include <vector>
#include <string>
#include <omp.h>

// 工作分配结构体
typedef struct WorkDivider {
    int num_threads;
    int tid;
    int num_numas;
    int threads_per_numa;
    int my_numa;
    int tid_in_numa;
} WorkDivider;

// 工作范围结构体:单numa和多numa
typedef struct SingleNumaWorkRange {
    int begin_thread;
    int end_thread;
    int work_per_thread;
} SingleNumaWorkRange;

typedef struct MultiNumaWorkRange {
    int begin_numa;
    int end_numa;
    int work_per_numa;
    int begin_thread;
    int end_thread;
    int work_per_thread;
} MultiNumaWorkRange;

// 全局变量声明
extern std::vector<int> cpu_ids;
extern bool i8mm_flag;

// 字符串工具函数
std::vector<std::string> split_str(const std::string& str, char delimiter);

// 线程管理函数
int get_total_thread_num();

// CPU亲和性相关函数
std::vector<int> parse_sequence(const std::string& input);
void get_affinity_cpus(std::vector<int>& cpu_ids);
void init_process_affinity();

// i8mm指令集检测
bool detect_i8mm_support();
bool init_i8mm_flag();

// 工作分配函数
void init_work_divider(WorkDivider* divider, int numas);
void divide_all_work(const WorkDivider* divider, int total_workitems, SingleNumaWorkRange* pstSingleRange);
void divide_work_first_numa(const WorkDivider* divider, int total_workitems, SingleNumaWorkRange* pstSingleRange);
void divide_work_all_numas(const WorkDivider* divider, int total_workitems, MultiNumaWorkRange* pstNulRange); 