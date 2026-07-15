# # # # import os
# # # # import json
# # # # import re
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt
# # # # from data.data_loader import DataLoader
# # # #
# # # # # ==========================================
# # # # # 学术图表全局设置 (SCI 顶刊级排版规范)
# # # # # ==========================================
# # # # plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
# # # # plt.rcParams['axes.unicode_minus'] = False
# # # # plt.rcParams['savefig.dpi'] = 600
# # # #
# # # # DB_CONFIG = {
# # # #     'host': 'localhost',
# # # #     'port': 3306,
# # # #     'user': 'root',
# # # #     'password': os.getenv('DB_PASSWORD', ''),
# # # #     'database': 'test3'
# # # # }
# # # #
# # # # # --- 核心配色方案 (终审级微调版) ---
# # # # COLOR_SHARED_BG = '#F5B7B1'  # 【微调】加实加深的浅砖红，缩放/投影更清晰
# # # # COLOR_NORMAL_BG = '#EBF5FB'  # 【微调】微微加深的浅蓝，与白底划清界限
# # # # COLOR_EMPTY_BG = '#FFFFFF'  # 纯白
# # # # COLOR_HEADER_BG = '#F2F4F4'  # 【微调】减淡的表头灰，消除顶部压迫感
# # # # COLOR_GRID_LINE = '#D5D8DC'  # 浅灰网格线
# # # # COLOR_TEXT_MAIN = '#2C3E50'  # 深炭黑文字
# # # #
# # # #
# # # # def format_course_name(name):
# # # #     """【新增】智能处理超长课程名，缓解机械班横向拥挤"""
# # # #     if not name: return ""
# # # #     # 去除常见的冗余前缀(如存在)或进行强制折行
# # # #     name = str(name).strip()
# # # #     if len(name) > 8:
# # # #         # 如果太长且没有换行，尝试在中间偏后的位置加换行符，或者直接截断
# # # #         # 简单策略：超过10个中文字符，强行中间换行
# # # #         if '\n' not in name and len(name) >= 10:
# # # #             mid = len(name) // 2
# # # #             name = name[:mid] + '\n' + name[mid:]
# # # #     return name
# # # #
# # # #
# # # # def plot_4_classes_schedule(genes, data, output_dir):
# # # #     print("📊 正在执行【终审级微调版】4班级全景排版...")
# # # #
# # # #     target_names_hints = ["电气21-1", "电气21-2", "机械23-2", "机械23-4"]
# # # #     target_class_ids = []
# # # #     target_class_names = []
# # # #
# # # #     for hint in target_names_hints:
# # # #         for cid, cinfo in data.classes.items():
# # # #             if hint in cinfo.class_name and cid not in target_class_ids:
# # # #                 target_class_ids.append(cid)
# # # #                 target_class_names.append(cinfo.class_name)
# # # #                 break
# # # #
# # # #     if len(target_class_ids) < 4:
# # # #         for cid, cinfo in data.classes.items():
# # # #             if len(target_class_ids) >= 4: break
# # # #             if cid not in target_class_ids:
# # # #                 target_class_ids.append(cid)
# # # #                 target_class_names.append(cinfo.class_name)
# # # #
# # # #     scheds = {cid: [[None for _ in range(5)] for _ in range(5)] for cid in target_class_ids}
# # # #
# # # #     for task_id_str, gene in genes.items():
# # # #         task_id = int(task_id_str)
# # # #         if task_id not in data.tasks: continue
# # # #         task = data.tasks[task_id]
# # # #
# # # #         if task.class_id in target_class_ids:
# # # #             slot = gene >> 16
# # # #             room_id = gene & 0xFFFF
# # # #             day, period = slot // 5, slot % 5
# # # #
# # # #             if 0 <= day < 5 and 0 <= period < 5:
# # # #                 t_id = str(task.teacher_id).strip()
# # # #                 t_name = data.teachers[t_id].teacher_name if t_id in data.teachers else f"{t_id}"
# # # #
# # # #                 r_key = str(room_id).zfill(6) if str(room_id) not in data.classrooms else str(room_id)
# # # #                 r_name = data.classrooms[r_key].classroom_name if r_key in data.classrooms else f"{room_id}"
# # # #
# # # #                 scheds[task.class_id][day][period] = {
# # # #                     'course': task.course_class_name,
# # # #                     'teacher_name': t_name,
# # # #                     'room_name': r_name,
# # # #                     'is_shared': False
# # # #                 }
# # # #
# # # #     # 结果导向判定 (同课程、同教师、同教室)
# # # #     for d in range(5):
# # # #         for p in range(5):
# # # #             sig_counts = {}
# # # #             for cid in target_class_ids:
# # # #                 item = scheds[cid][d][p]
# # # #                 if item:
# # # #                     sig = f"{item['course']}_{item['teacher_name']}_{item['room_name']}"
# # # #                     sig_counts[sig] = sig_counts.get(sig, 0) + 1
# # # #
# # # #             for cid in target_class_ids:
# # # #                 item = scheds[cid][d][p]
# # # #                 if item:
# # # #                     sig = f"{item['course']}_{item['teacher_name']}_{item['room_name']}"
# # # #                     if sig_counts[sig] >= 2:
# # # #                         item['is_shared'] = True
# # # #
# # # #     # ==========================================
# # # #     # 绘图渲染区 (终审微调)
# # # #     # ==========================================
# # # #     fig, axes = plt.subplots(2, 2, figsize=(16, 11))
# # # #     axes = axes.flatten()
# # # #
# # # #     days_str = ['周一', '周二', '周三', '周四', '周五']
# # # #     periods_str = ['第1-2节', '第3-4节', '第5-6节', '第7-8节', '第9-10节']
# # # #
# # # #     for idx, cid in enumerate(target_class_ids):
# # # #         ax = axes[idx]
# # # #         name = target_class_names[idx]
# # # #         sched = scheds[cid]
# # # #
# # # #         ax.axis('off')
# # # #         cell_text = [['' for _ in range(5)] for _ in range(5)]
# # # #         cell_colors = [[COLOR_EMPTY_BG for _ in range(5)] for _ in range(5)]
# # # #
# # # #         for d in range(5):
# # # #             for p in range(5):
# # # #                 item = sched[d][p]
# # # #                 if item:
# # # #                     # 【微调】：智能处理长课程名
# # # #                     c_name_formatted = format_course_name(item['course'])
# # # #                     cell_text[p][d] = f"《{c_name_formatted}》\n{item['teacher_name']}\n{item['room_name']}"
# # # #                     cell_colors[p][d] = COLOR_SHARED_BG if item['is_shared'] else COLOR_NORMAL_BG
# # # #
# # # #         table = ax.table(cellText=cell_text, rowLabels=periods_str, colLabels=days_str,
# # # #                          cellColours=cell_colors, loc='center', cellLoc='center')
# # # #
# # # #         table.auto_set_font_size(False)
# # # #         table.set_fontsize(9.8)  # 【微调】：字号降至 9.8，释放横向空间
# # # #         table.scale(1, 4.3)  # 略微拉高适应可能的折行
# # # #
# # # #         for (row, col), cell in table.get_celld().items():
# # # #             cell.set_edgecolor(COLOR_GRID_LINE)
# # # #             cell.set_linewidth(0.8)
# # # #             cell.set_text_props(color=COLOR_TEXT_MAIN)
# # # #
# # # #             if row == 0 or col == -1:
# # # #                 cell.set_facecolor(COLOR_HEADER_BG)
# # # #                 cell.set_text_props(weight='bold', fontsize=11)
# # # #             elif col >= 0 and row > 0 and sched[col][row - 1] and sched[col][row - 1]['is_shared']:
# # # #                 cell.set_text_props(weight='bold', color='#17202A')
# # # #             elif col >= 0 and row > 0 and not sched[col][row - 1]:
# # # #                 cell.set_edgecolor('#E5E7E9')
# # # #
# # # #                 # 【微调】：pad 增至 16，拉开标题与表格间距
# # # #         ax.set_title(f"行政班：{name}", fontsize=13, weight='normal', color=COLOR_TEXT_MAIN, pad=16)
# # # #
# # # #     fig.suptitle('图3-xx 本文算法跨专业合班时空同构性全景图',
# # # #                  fontsize=16, weight='bold', color='#1C2833', y=0.97)
# # # #
# # # #     plt.subplots_adjust(wspace=0.15, hspace=0.25)
# # # #     plt.tight_layout(rect=[0, 0, 1, 0.94])
# # # #     plt.savefig(os.path.join(output_dir, 'Fig1_Result_Oriented_4Classes_Final.png'))
# # # #     plt.close()
# # # #     print("✨ 图1 (终审级排版) 绘制完美收工！可以直接送审了！")
# # # #
# # # #
# # # # def main():
# # # #     path_ablation = "results/<experiment>/L1_Large/best_schedules.json"
# # # #     output_dir = "paper_plots"
# # # #     os.makedirs(output_dir, exist_ok=True)
# # # #
# # # #     if not os.path.exists(path_ablation):
# # # #         print(f"❌ 找不到 JSON 文件: {path_ablation}")
# # # #         return
# # # #
# # # #     with open(path_ablation, 'r', encoding='utf-8') as f:
# # # #         schedules = json.load(f)
# # # #
# # # #     try:
# # # #         print("🔗 正在连接数据库拉取最新实例数据...")
# # # #         loader = DataLoader('L1_Large', DB_CONFIG)
# # # #         data = loader.load()
# # # #
# # # #         algo_key = next((k for k in schedules if "TrueCDM" in k), None)
# # # #         if algo_key:
# # # #             plot_4_classes_schedule(schedules[algo_key]['schedule'], data, output_dir)
# # # #             print(f"\n🎉 完美！绝杀图表已导出至:\n📁 {output_dir}")
# # # #         else:
# # # #             print("⚠️ 未找到包含 TrueCDM 的算法数据。")
# # # #     except Exception as e:
# # # #         print(f"⚠️ 绘图失败: {e}")
# # # #
# # # #
# # # # if __name__ == "__main__":
# # # #     main()
# # #
# # # import os
# # # import json
# # # import re
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from data.data_loader import DataLoader
# # #
# # # # ==========================================
# # # # 学术图表全局设置 (SCI 顶刊级排版规范)
# # # # ==========================================
# # # plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
# # # plt.rcParams['axes.unicode_minus'] = False
# # # plt.rcParams['savefig.dpi'] = 600
# # #
# # # DB_CONFIG = {
# # #     'host': 'localhost',
# # #     'port': 3306,
# # #     'user': 'root',
# # #     'password': os.getenv('DB_PASSWORD', ''),
# # #     'database': 'test3'
# # # }
# # #
# # # # --- 核心配色方案 (绿色最高优先级) ---
# # # COLOR_FIXED_BG = '#D5F5E3'  # 浅青绿：固定课程 (优先级1，含固定合班)
# # # COLOR_SHARED_BG = '#F5B7B1'  # 浅砖红：普通合班共享 (优先级2)
# # # COLOR_NORMAL_BG = '#EBF5FB'  # 浅灰蓝：普通独立课程 (优先级3)
# # # COLOR_EMPTY_BG = '#FFFFFF'  # 纯白：空白时段
# # # COLOR_HEADER_BG = '#F2F4F4'  # 浅灰：表头
# # # COLOR_GRID_LINE = '#D5D8DC'  # 浅灰：网格线
# # # COLOR_TEXT_MAIN = '#2C3E50'  # 深炭黑：主文字
# # #
# # #
# # # def format_course_name(name):
# # #     """智能处理超长课程名，缓解横向拥挤"""
# # #     if not name: return ""
# # #     name = str(name).strip()
# # #     if len(name) > 8:
# # #         if '\n' not in name and len(name) >= 10:
# # #             mid = len(name) // 2
# # #             name = name[:mid] + '\n' + name[mid:]
# # #     return name
# # #
# # #
# # # def plot_6_classes_schedule(genes, data, output_dir):
# # #     print("📊 正在执行【固定优先(绿) + 3行2列布局】排版...")
# # #
# # #     # 1. 锁定指定的 6 个班级
# # #     target_names_hints = ["电气21-1", "电气21-2", "2022110201", "20122110202", "2021020302", "20221020301"]
# # #     target_class_ids = []
# # #     target_class_names = []
# # #
# # #     for hint in target_names_hints:
# # #         for cid, cinfo in data.classes.items():
# # #             if (hint in cinfo.class_name or hint in str(cid)) and cid not in target_class_ids:
# # #                 target_class_ids.append(cid)
# # #                 target_class_names.append(cinfo.class_name)
# # #                 break
# # #
# # #     # 自动补齐机制
# # #     if len(target_class_ids) < 6:
# # #         print(f"⚠️ 仅找到 {len(target_class_ids)} 个指定班级，正在自动补齐至 6 个...")
# # #         for cid, cinfo in data.classes.items():
# # #             if len(target_class_ids) >= 6: break
# # #             if cid not in target_class_ids:
# # #                 target_class_ids.append(cid)
# # #                 target_class_names.append(cinfo.class_name)
# # #
# # #     scheds = {cid: [[None for _ in range(5)] for _ in range(5)] for cid in target_class_ids[:6]}
# # #
# # #     for task_id_str, gene in genes.items():
# # #         task_id = int(task_id_str)
# # #         if task_id not in data.tasks: continue
# # #         task = data.tasks[task_id]
# # #
# # #         if task.class_id in target_class_ids[:6]:
# # #             slot = gene >> 16
# # #             room_id = gene & 0xFFFF
# # #             day, period = slot // 5, slot % 5
# # #
# # #             if 0 <= day < 5 and 0 <= period < 5:
# # #                 t_id = str(task.teacher_id).strip()
# # #                 t_name = data.teachers[t_id].teacher_name if t_id in data.teachers else f"{t_id}"
# # #
# # #                 r_key = str(room_id).zfill(6) if str(room_id) not in data.classrooms else str(room_id)
# # #                 r_name = data.classrooms[r_key].classroom_name if r_key in data.classrooms else f"{room_id}"
# # #
# # #                 is_fixed = getattr(task, 'is_fixed', False)
# # #
# # #                 scheds[task.class_id][day][period] = {
# # #                     'course': task.course_class_name,
# # #                     'teacher_name': t_name,
# # #                     'room_name': r_name,
# # #                     'is_shared': False,
# # #                     'is_fixed': is_fixed
# # #                 }
# # #
# # #     # 结果导向合班判定
# # #     for d in range(5):
# # #         for p in range(5):
# # #             sig_counts = {}
# # #             for cid in target_class_ids[:6]:
# # #                 item = scheds[cid][d][p]
# # #                 if item:
# # #                     sig = f"{item['course']}_{item['teacher_name']}_{item['room_name']}"
# # #                     sig_counts[sig] = sig_counts.get(sig, 0) + 1
# # #
# # #             for cid in target_class_ids[:6]:
# # #                 item = scheds[cid][d][p]
# # #                 if item:
# # #                     sig = f"{item['course']}_{item['teacher_name']}_{item['room_name']}"
# # #                     if sig_counts[sig] >= 2:
# # #                         item['is_shared'] = True
# # #
# # #     # ==========================================
# # #     # 绘图渲染区
# # #     # ==========================================
# # #     fig, axes = plt.subplots(3, 2, figsize=(16, 18))
# # #     axes = axes.flatten()
# # #
# # #     days_str = ['周一', '周二', '周三', '周四', '周五']
# # #     periods_str = ['第1-2节', '第3-4节', '第5-6节', '第7-8节', '第9-10节']
# # #
# # #     for idx, cid in enumerate(target_class_ids[:6]):
# # #         ax = axes[idx]
# # #         name = target_class_names[idx]
# # #         sched = scheds[cid]
# # #
# # #         ax.axis('off')
# # #         cell_text = [['' for _ in range(5)] for _ in range(5)]
# # #         cell_colors = [[COLOR_EMPTY_BG for _ in range(5)] for _ in range(5)]
# # #
# # #         for d in range(5):
# # #             for p in range(5):
# # #                 item = sched[d][p]
# # #                 if item:
# # #                     c_name_formatted = format_course_name(item['course'])
# # #                     prefix = "[固]" if item['is_fixed'] else ""
# # #                     cell_text[p][d] = f"{prefix}《{c_name_formatted}》\n{item['teacher_name']}\n{item['room_name']}"
# # #
# # #                     if item['is_fixed']:
# # #                         cell_colors[p][d] = COLOR_FIXED_BG
# # #                     elif item['is_shared']:
# # #                         cell_colors[p][d] = COLOR_SHARED_BG
# # #                     else:
# # #                         cell_colors[p][d] = COLOR_NORMAL_BG
# # #
# # #         table = ax.table(cellText=cell_text, rowLabels=periods_str, colLabels=days_str,
# # #                          cellColours=cell_colors, loc='center', cellLoc='center')
# # #
# # #         table.auto_set_font_size(False)
# # #         table.set_fontsize(10.5)
# # #         table.scale(1, 4.2)
# # #
# # #         for (row, col), cell in table.get_celld().items():
# # #             cell.set_edgecolor(COLOR_GRID_LINE)
# # #             cell.set_linewidth(0.8)
# # #             cell.set_text_props(color=COLOR_TEXT_MAIN)
# # #
# # #             if row == 0 or col == -1:
# # #                 cell.set_facecolor(COLOR_HEADER_BG)
# # #                 cell.set_text_props(weight='bold', fontsize=12)
# # #             elif col >= 0 and row > 0 and sched[col][row - 1]:
# # #                 item = sched[col][row - 1]
# # #                 if item['is_shared'] or item['is_fixed']:
# # #                     cell.set_text_props(weight='bold', color='#17202A')
# # #             elif col >= 0 and row > 0 and not sched[col][row - 1]:
# # #                 cell.set_edgecolor('#E5E7E9')
# # #
# # #         ax.set_title(f"行政班：{name}", fontsize=15, weight='bold', color=COLOR_TEXT_MAIN, pad=18)
# # #
# # #     fig.suptitle('图3-xx 本文算法跨专业合班与固定课程时空调度全景图',
# # #                  fontsize=22, weight='bold', color='#1C2833', y=0.97)
# # #
# # #     plt.subplots_adjust(wspace=0.15, hspace=0.35)
# # #     plt.tight_layout(rect=[0, 0, 1, 0.94])
# # #
# # #     # ==========================================
# # #     # 【新增】：同时保存 PNG 和 SVG，并明确路径
# # #     # ==========================================
# # #     png_path = os.path.join(output_dir, 'Fig1_Result_Oriented_6Classes_Vertical.png')
# # #     svg_path = os.path.join(output_dir, 'Fig1_Result_Oriented_6Classes_Vertical.svg')
# # #
# # #     plt.savefig(png_path, dpi=600, format='png')
# # #     plt.savefig(svg_path, format='svg')  # 输出矢量图
# # #     plt.close()
# # #
# # #     print(f"✨ 图1 绘制完美收工！")
# # #     print(f"📄 已保存位图 (用于Word): {png_path}")
# # #     print(f"📄 已保存矢量图 (用于排版/修改): {svg_path}")
# # #
# # #
# # # def main():
# # #     # ==========================================
# # #     # 明确指定的绝对输入与输出路径
# # #     # ==========================================
# # #     path_ablation = "results/<experiment>/L1_Large/best_schedules.json"
# # #     output_dir = "paper_plots"
# # #
# # #     # 确保输出目录一定存在
# # #     os.makedirs(output_dir, exist_ok=True)
# # #
# # #     if not os.path.exists(path_ablation):
# # #         print(f"❌ 找不到 JSON 文件: {path_ablation}")
# # #         return
# # #
# # #     with open(path_ablation, 'r', encoding='utf-8') as f:
# # #         schedules = json.load(f)
# # #
# # #     try:
# # #         print("🔗 正在连接数据库拉取最新实例数据...")
# # #         loader = DataLoader('L1_Large', DB_CONFIG)
# # #         data = loader.load()
# # #
# # #         algo_key = next((k for k in schedules if "TrueCDM" in k), None)
# # #         if algo_key:
# # #             plot_6_classes_schedule(schedules[algo_key]['schedule'], data, output_dir)
# # #             print(f"\n🎉 完美！双格式图表已成功导出至目录:\n📁 {output_dir}")
# # #         else:
# # #             print("⚠️ 未找到包含 TrueCDM 的算法数据。")
# # #     except Exception as e:
# # #         print(f"⚠️ 绘图失败: {e}")
# # #
# # #
# # # if __name__ == "__main__":
# # #     main()
# #
# # import os
# # import json
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from data.data_loader import DataLoader
# #
# # # ==========================================
# # # 学术图表全局设置 (SCI 顶刊级排版规范)
# # # ==========================================
# # plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
# # plt.rcParams['axes.unicode_minus'] = False
# # plt.rcParams['savefig.dpi'] = 600
# #
# # DB_CONFIG = {
# #     'host': 'localhost',
# #     'port': 3306,
# #     'user': 'root',
# #     'password': os.getenv('DB_PASSWORD', ''),
# #     'database': 'test3'
# # }
# #
# # # --- 核心配色方案 (绿色最高优先级) ---
# # COLOR_FIXED_BG = '#D5F5E3'  # 浅青绿：固定课程 (优先级1)
# # COLOR_SHARED_BG = '#F5B7B1'  # 浅砖红：普通合班共享 (优先级2)
# # COLOR_NORMAL_BG = '#EBF5FB'  # 浅灰蓝：普通独立课程 (优先级3)
# # COLOR_EMPTY_BG = '#FFFFFF'  # 纯白：空白时段
# # COLOR_HEADER_BG = '#F2F4F4'  # 浅灰：表头
# # COLOR_GRID_LINE = '#D5D8DC'  # 浅灰：网格线
# # COLOR_TEXT_MAIN = '#2C3E50'  # 深炭黑：主文字
# #
# #
# # def format_course_name(name):
# #     """智能处理超长课程名，缓解横向拥挤"""
# #     if not name: return ""
# #     name = str(name).strip()
# #     if len(name) > 8:
# #         if '\n' not in name and len(name) >= 10:
# #             mid = len(name) // 2
# #             name = name[:mid] + '\n' + name[mid:]
# #     return name
# #
# #
# # def plot_4_classes_schedule(genes, data, output_dir):
# #     print("📊 正在执行【终极4班级 + 绿色固定优先 + 2x2完美布局】排版...")
# #
# #     # 1. 严格锁定您指定的 4 个目标
# #     target_names_hints = ["机械23-2", "机械23-4", "2022110201", "2022110202"]
# #     target_class_ids = []
# #     target_class_names = []
# #
# #     # 强制精准匹配逻辑
# #     for hint in target_names_hints:
# #         found = False
# #         for cid, cinfo in data.classes.items():
# #             if (hint in cinfo.class_name or hint == str(cid)) and cid not in target_class_ids:
# #                 target_class_ids.append(cid)
# #                 target_class_names.append(cinfo.class_name)
# #                 found = True
# #                 break
# #
# #         # 强制兜底：如果数据库没查到名字，强制把 ID 塞进去画图！
# #         if not found:
# #             print(f"⚠️ 提示：未查到 '{hint}' 的名字，已强制按ID提取专属排课数据！")
# #             force_id = int(hint) if hint.isdigit() else hint
# #             target_class_ids.append(force_id)
# #             target_class_names.append(f"班级号 {hint}")
# #
# #     print(f"🎯 最终入列展示的 4 个班级（严格锁定）: {target_class_names}")
# #
# #     scheds = {cid: [[None for _ in range(5)] for _ in range(5)] for cid in target_class_ids}
# #
# #     for task_id_str, gene in genes.items():
# #         task_id = int(task_id_str)
# #         if task_id not in data.tasks: continue
# #         task = data.tasks[task_id]
# #
# #         if task.class_id in target_class_ids:
# #             slot = gene >> 16
# #             room_id = gene & 0xFFFF
# #             day, period = slot // 5, slot % 5
# #
# #             if 0 <= day < 5 and 0 <= period < 5:
# #                 t_id = str(task.teacher_id).strip()
# #                 t_name = data.teachers[t_id].teacher_name if t_id in data.teachers else f"{t_id}"
# #
# #                 r_key = str(room_id).zfill(6) if str(room_id) not in data.classrooms else str(room_id)
# #                 r_name = data.classrooms[r_key].classroom_name if r_key in data.classrooms else f"{room_id}"
# #
# #                 is_fixed = getattr(task, 'is_fixed', False)
# #
# #                 scheds[task.class_id][day][period] = {
# #                     'course': task.course_class_name,
# #                     'teacher_name': t_name,
# #                     'room_name': r_name,
# #                     'is_shared': False,
# #                     'is_fixed': is_fixed
# #                 }
# #
# #     # 结果导向合班判定
# #     for d in range(5):
# #         for p in range(5):
# #             sig_counts = {}
# #             for cid in target_class_ids:
# #                 item = scheds[cid][d][p]
# #                 if item:
# #                     sig = f"{item['course']}_{item['teacher_name']}_{item['room_name']}"
# #                     sig_counts[sig] = sig_counts.get(sig, 0) + 1
# #
# #             for cid in target_class_ids:
# #                 item = scheds[cid][d][p]
# #                 if item:
# #                     sig = f"{item['course']}_{item['teacher_name']}_{item['room_name']}"
# #                     if sig_counts[sig] >= 2:
# #                         item['is_shared'] = True
# #
# #     # ==========================================
# #     # 绘图渲染区 (2行 x 2列 完美矩阵布局)
# #     # ==========================================
# #     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# #     axes = axes.flatten()
# #
# #     days_str = ['周一', '周二', '周三', '周四', '周五']
# #     periods_str = ['第1-2节', '第3-4节', '第5-6节', '第7-8节', '第9-10节']
# #
# #     for idx, cid in enumerate(target_class_ids):
# #         ax = axes[idx]
# #         name = target_class_names[idx]
# #         sched = scheds[cid]
# #
# #         ax.axis('off')
# #         cell_text = [['' for _ in range(5)] for _ in range(5)]
# #         cell_colors = [[COLOR_EMPTY_BG for _ in range(5)] for _ in range(5)]
# #
# #         for d in range(5):
# #             for p in range(5):
# #                 item = sched[d][p]
# #                 if item:
# #                     c_name_formatted = format_course_name(item['course'])
# #                     prefix = "[固]" if item['is_fixed'] else ""
# #                     cell_text[p][d] = f"{prefix}《{c_name_formatted}》\n{item['teacher_name']}\n{item['room_name']}"
# #
# #                     # 优先级：固定绿 > 合班红 > 普通蓝
# #                     if item['is_fixed']:
# #                         cell_colors[p][d] = COLOR_FIXED_BG
# #                     elif item['is_shared']:
# #                         cell_colors[p][d] = COLOR_SHARED_BG
# #                     else:
# #                         cell_colors[p][d] = COLOR_NORMAL_BG
# #
# #         table = ax.table(cellText=cell_text, rowLabels=periods_str, colLabels=days_str,
# #                          cellColours=cell_colors, loc='center', cellLoc='center')
# #
# #         table.auto_set_font_size(False)
# #         table.set_fontsize(10.5)
# #         table.scale(1, 4.3)
# #
# #         for (row, col), cell in table.get_celld().items():
# #             cell.set_edgecolor(COLOR_GRID_LINE)
# #             cell.set_linewidth(0.8)
# #             cell.set_text_props(color=COLOR_TEXT_MAIN)
# #
# #             if row == 0 or col == -1:
# #                 cell.set_facecolor(COLOR_HEADER_BG)
# #                 cell.set_text_props(weight='bold', fontsize=12)
# #             elif col >= 0 and row > 0 and sched[col][row - 1]:
# #                 item = sched[col][row - 1]
# #                 if item['is_shared'] or item['is_fixed']:
# #                     cell.set_text_props(weight='bold', color='#17202A')
# #             elif col >= 0 and row > 0 and not sched[col][row - 1]:
# #                 cell.set_edgecolor('#E5E7E9')
# #
# #         ax.set_title(f"行政班：{name}", fontsize=15, weight='bold', color=COLOR_TEXT_MAIN, pad=18)
# #
# #     fig.suptitle('合班排课表',
# #                  fontsize=22, weight='bold', color='#1C2833', y=0.97)
# #
# #     # 释放上下呼吸空间
# #     plt.subplots_adjust(wspace=0.15, hspace=0.3)
# #     plt.tight_layout(rect=[0, 0, 1, 0.94])
# #
# #     # 双格式高清导出
# #     png_path = os.path.join(output_dir, 'Fig1_Result_Oriented_4Classes_Final.png')
# #     svg_path = os.path.join(output_dir, 'Fig1_Result_Oriented_4Classes_Final.svg')
# #
# #     plt.savefig(png_path, dpi=600, format='png')
# #     plt.savefig(svg_path, format='svg')
# #     plt.close()
# #
# #     print(f"✨ 图1 (终极4宫格版) 绘制完美收工！")
# #     print(f"📄 位图保存至: {png_path}")
# #     print(f"📄 矢量图保存至: {svg_path}")
# #
# #
# # def main():
# #     path_ablation = "results/<experiment>/L1_Large/best_schedules.json"
# #     output_dir = "paper_plots"
# #
# #     os.makedirs(output_dir, exist_ok=True)
# #
# #     if not os.path.exists(path_ablation):
# #         print(f"❌ 找不到 JSON 文件: {path_ablation}")
# #         return
# #
# #     with open(path_ablation, 'r', encoding='utf-8') as f:
# #         schedules = json.load(f)
# #
# #     try:
# #         print("🔗 正在连接数据库拉取最新实例数据...")
# #         loader = DataLoader('L1_Large', DB_CONFIG)
# #         data = loader.load()
# #
# #         algo_key = next((k for k in schedules if "TrueCDM" in k), None)
# #         if algo_key:
# #             plot_4_classes_schedule(schedules[algo_key]['schedule'], data, output_dir)
# #             print(f"\n🎉 完美！双格式图表已成功导出至目录:\n📁 {output_dir}")
# #         else:
# #             print("⚠️ 未找到包含 TrueCDM 的算法数据。")
# #     except Exception as e:
# #         print(f"⚠️ 绘图失败: {e}")
# #
# #
# # if __name__ == "__main__":
# #     main()
#
# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches  # 【新增】用于绘制图例图标
# from data.data_loader import DataLoader
#
# # ==========================================
# # 学术图表全局设置 (SCI 顶刊级排版规范)
# # ==========================================
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['savefig.dpi'] = 600
#
# DB_CONFIG = {
#     'host': 'localhost',
#     'port': 3306,
#     'user': 'root',
#     'password': os.getenv('DB_PASSWORD', ''),
#     'database': 'test3'
# }
#
# # --- 核心配色方案 (绿色最高优先级) ---
# COLOR_FIXED_BG = '#D5F5E3'  # 浅青绿：固定课程 (优先级1)
# COLOR_SHARED_BG = '#F5B7B1'  # 浅砖红：普通合班共享 (优先级2)
# COLOR_NORMAL_BG = '#EBF5FB'  # 浅灰蓝：普通独立课程 (优先级3)
# COLOR_EMPTY_BG = '#FFFFFF'  # 纯白：空白时段
# COLOR_HEADER_BG = '#F2F4F4'  # 浅灰：表头
# COLOR_GRID_LINE = '#D5D8DC'  # 浅灰：网格线
# COLOR_TEXT_MAIN = '#2C3E50'  # 深炭黑：主文字
#
#
# def format_course_name(name):
#     """智能处理超长课程名，缓解横向拥挤"""
#     if not name: return ""
#     name = str(name).strip()
#     if len(name) > 8:
#         if '\n' not in name and len(name) >= 10:
#             mid = len(name) // 2
#             name = name[:mid] + '\n' + name[mid:]
#     return name
#
#
# def plot_4_classes_schedule(genes, data, output_dir):
#     print("📊 正在执行【终极4班级 + 顶部图例图标 + 2x2完美布局】排版...")
#
#     # 1. 严格锁定您指定的 4 个目标
#     target_names_hints = ["电气23-2", "电气23-1", "机械23-2", "机械23-4"]
#     target_class_ids = []
#     target_class_names = []
#
#     for hint in target_names_hints:
#         found = False
#         for cid, cinfo in data.classes.items():
#             if (hint in cinfo.class_name or hint == str(cid)) and cid not in target_class_ids:
#                 target_class_ids.append(cid)
#                 target_class_names.append(cinfo.class_name)
#                 found = True
#                 break
#
#         if not found:
#             print(f"⚠️ 提示：未查到 '{hint}' 的名字，已强制按ID提取专属排课数据！")
#             force_id = int(hint) if hint.isdigit() else hint
#             target_class_ids.append(force_id)
#             target_class_names.append(f"班级号 {hint}")
#
#     scheds = {cid: [[None for _ in range(5)] for _ in range(5)] for cid in target_class_ids}
#
#     for task_id_str, gene in genes.items():
#         task_id = int(task_id_str)
#         if task_id not in data.tasks: continue
#         task = data.tasks[task_id]
#
#         if task.class_id in target_class_ids:
#             slot = gene >> 16
#             room_id = gene & 0xFFFF
#             day, period = slot // 5, slot % 5
#
#             if 0 <= day < 5 and 0 <= period < 5:
#                 t_id = str(task.teacher_id).strip()
#                 t_name = data.teachers[t_id].teacher_name if t_id in data.teachers else f"{t_id}"
#
#                 r_key = str(room_id).zfill(6) if str(room_id) not in data.classrooms else str(room_id)
#                 r_name = data.classrooms[r_key].classroom_name if r_key in data.classrooms else f"{room_id}"
#
#                 is_fixed = getattr(task, 'is_fixed', False)
#
#                 scheds[task.class_id][day][period] = {
#                     'course': task.course_class_name,
#                     'teacher_name': t_name,
#                     'room_name': r_name,
#                     'is_shared': False,
#                     'is_fixed': is_fixed
#                 }
#
#     # 结果导向合班判定
#     for d in range(5):
#         for p in range(5):
#             sig_counts = {}
#             for cid in target_class_ids:
#                 item = scheds[cid][d][p]
#                 if item:
#                     sig = f"{item['course']}_{item['teacher_name']}_{item['room_name']}"
#                     sig_counts[sig] = sig_counts.get(sig, 0) + 1
#
#             for cid in target_class_ids:
#                 item = scheds[cid][d][p]
#                 if item:
#                     sig = f"{item['course']}_{item['teacher_name']}_{item['room_name']}"
#                     if sig_counts[sig] >= 2:
#                         item['is_shared'] = True
#
#     # ==========================================
#     # 绘图渲染区
#     # ==========================================
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#     axes = axes.flatten()
#
#     days_str = ['周一', '周二', '周三', '周四', '周五']
#     periods_str = ['第1-2节', '第3-4节', '第5-6节', '第7-8节', '第9-10节']
#
#     for idx, cid in enumerate(target_class_ids):
#         ax = axes[idx]
#         name = target_class_names[idx]
#         sched = scheds[cid]
#
#         ax.axis('off')
#         cell_text = [['' for _ in range(5)] for _ in range(5)]
#         cell_colors = [[COLOR_EMPTY_BG for _ in range(5)] for _ in range(5)]
#
#         for d in range(5):
#             for p in range(5):
#                 item = sched[d][p]
#                 if item:
#                     c_name_formatted = format_course_name(item['course'])
#                     prefix = "[固]" if item['is_fixed'] else ""
#                     cell_text[p][d] = f"{prefix}《{c_name_formatted}》\n{item['teacher_name']}\n{item['room_name']}"
#
#                     if item['is_fixed']:
#                         cell_colors[p][d] = COLOR_FIXED_BG
#                     elif item['is_shared']:
#                         cell_colors[p][d] = COLOR_SHARED_BG
#                     else:
#                         cell_colors[p][d] = COLOR_NORMAL_BG
#
#         table = ax.table(cellText=cell_text, rowLabels=periods_str, colLabels=days_str,
#                          cellColours=cell_colors, loc='center', cellLoc='center')
#
#         table.auto_set_font_size(False)
#         table.set_fontsize(10.5)
#         table.scale(1, 4.3)
#
#         for (row, col), cell in table.get_celld().items():
#             cell.set_edgecolor(COLOR_GRID_LINE)
#             cell.set_linewidth(0.8)
#             cell.set_text_props(color=COLOR_TEXT_MAIN)
#
#             if row == 0 or col == -1:
#                 cell.set_facecolor(COLOR_HEADER_BG)
#                 cell.set_text_props(weight='bold', fontsize=12)
#             elif col >= 0 and row > 0 and sched[col][row - 1]:
#                 item = sched[col][row - 1]
#                 if item['is_shared'] or item['is_fixed']:
#                     cell.set_text_props(weight='bold', color='#17202A')
#             elif col >= 0 and row > 0 and not sched[col][row - 1]:
#                 cell.set_edgecolor('#E5E7E9')
#
#         ax.set_title(f"行政班：{name}", fontsize=15, weight='bold', color=COLOR_TEXT_MAIN, pad=18)
#
#     # ==========================================
#     # 【核心新增】：全局大标题与颜色图标(Legend)说明
#     # ==========================================
#     fig.suptitle('本文算法合班与固定课程示意图',
#                  fontsize=22, weight='bold', color='#1C2833', y=0.98)
#
#     # 绘制自定义颜色图例 (Legend Icons)
#     legend_elements = [
#         mpatches.Patch(facecolor=COLOR_FIXED_BG, edgecolor='#AAB7B8', linewidth=1, label='固定预排课程 (硬约束优先)'),
#         mpatches.Patch(facecolor=COLOR_SHARED_BG, edgecolor='#AAB7B8', linewidth=1, label='合班共享课程 (物理时空同步)'),
#         mpatches.Patch(facecolor=COLOR_NORMAL_BG, edgecolor='#AAB7B8', linewidth=1, label='普通独立课程 (自主寻优)')
#     ]
#
#     # 将图标横向排开，放置在大标题的正下方、课表的正上方
#     fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.94),
#                ncol=3, fontsize=13, frameon=False, columnspacing=3.0)
#
#     # 释放上下呼吸空间，给图例留出位置 (顶部压低至 0.90)
#     plt.subplots_adjust(wspace=0.15, hspace=0.3)
#     plt.tight_layout(rect=[0, 0, 1, 0.90])
#
#     # 双格式高清导出
#     png_path = os.path.join(output_dir, 'Fig1_Result_Oriented_4Classes_Final.png')
#     svg_path = os.path.join(output_dir, 'Fig1_Result_Oriented_4Classes_Final.svg')
#
#     plt.savefig(png_path, dpi=600, format='png')
#     plt.savefig(svg_path, format='svg')
#     plt.close()
#
#     print(f"✨ 图1 (含顶部精美图例) 绘制完美收工！")
#     print(f"📄 位图保存至: {png_path}")
#     print(f"📄 矢量图保存至: {svg_path}")
#
#
# def main():
#     path_ablation = "results/<experiment>/L1_Large/best_schedules.json"
#     output_dir = "paper_plots"
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     if not os.path.exists(path_ablation):
#         print(f"❌ 找不到 JSON 文件: {path_ablation}")
#         return
#
#     with open(path_ablation, 'r', encoding='utf-8') as f:
#         schedules = json.load(f)
#
#     try:
#         print("🔗 正在连接数据库拉取最新实例数据...")
#         loader = DataLoader('L1_Large', DB_CONFIG)
#         data = loader.load()
#
#         algo_key = next((k for k in schedules if "TrueCDM" in k), None)
#         if algo_key:
#             plot_4_classes_schedule(schedules[algo_key]['schedule'], data, output_dir)
#             print(f"\n🎉 完美！双格式图表已成功导出至目录:\n📁 {output_dir}")
#         else:
#             print("⚠️ 未找到包含 TrueCDM 的算法数据。")
#     except Exception as e:
#         print(f"⚠️ 绘图失败: {e}")
#
#
# if __name__ == "__main__":
#     main()

import argparse
import os
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # 用于绘制图例图标
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import DataLoader

# ==========================================
# 学术图表全局设置 (SCI 顶刊级排版规范)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 600

DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': os.getenv('DB_PASSWORD', ''),
    'database': 'test3'
}

# --- 核心配色方案 (绿色最高优先级) ---
COLOR_FIXED_BG = '#D5F5E3'  # 浅青绿：固定课程 (优先级1)
COLOR_SHARED_BG = '#F5B7B1'  # 浅砖红：普通合班共享 (优先级2)
COLOR_NORMAL_BG = '#EBF5FB'  # 浅灰蓝：普通独立课程 (优先级3)
COLOR_EMPTY_BG = '#FFFFFF'  # 纯白：空白时段
COLOR_HEADER_BG = '#F2F4F4'  # 浅灰：表头
COLOR_GRID_LINE = '#D5D8DC'  # 浅灰：网格线
COLOR_TEXT_MAIN = '#2C3E50'  # 深炭黑：主文字


def format_course_name(name):
    """智能处理超长课程名，缓解横向拥挤"""
    if not name: return ""
    name = str(name).strip()
    if len(name) > 8:
        if '\n' not in name and len(name) >= 10:
            mid = len(name) // 2
            name = name[:mid] + '\n' + name[mid:]
    return name


def plot_4_classes_schedule(genes, data, output_dir):
    print("📊 正在执行【终极4班级 + 顶部图例图标 + 2x2完美布局】排版...")

    # 1. 严格锁定您指定的 4 个目标
    target_names_hints = ["电气23-2", "电气23-1", "机械23-2", "机械23-4"]
    target_class_ids = []
    target_class_names = []

    for hint in target_names_hints:
        found = False
        for cid, cinfo in data.classes.items():
            if (hint in cinfo.class_name or hint == str(cid)) and cid not in target_class_ids:
                target_class_ids.append(cid)
                target_class_names.append(cinfo.class_name)
                found = True
                break

        if not found:
            print(f"⚠️ 提示：未查到 '{hint}' 的名字，已强制按ID提取专属排课数据！")
            force_id = int(hint) if hint.isdigit() else hint
            target_class_ids.append(force_id)
            target_class_names.append(f"班级号 {hint}")

    scheds = {cid: [[None for _ in range(5)] for _ in range(5)] for cid in target_class_ids}

    for task_id_str, gene in genes.items():
        task_id = int(task_id_str)
        if task_id not in data.tasks: continue
        task = data.tasks[task_id]

        if task.class_id in target_class_ids:
            slot = gene >> 16
            room_id = gene & 0xFFFF
            day, period = slot // 5, slot % 5

            if 0 <= day < 5 and 0 <= period < 5:
                t_id = str(task.teacher_id).strip()
                t_name = data.teachers[t_id].teacher_name if t_id in data.teachers else f"{t_id}"

                r_key = str(room_id).zfill(6) if str(room_id) not in data.classrooms else str(room_id)
                r_name = data.classrooms[r_key].classroom_name if r_key in data.classrooms else f"{room_id}"

                is_fixed = getattr(task, 'is_fixed', False)

                scheds[task.class_id][day][period] = {
                    'course': task.course_class_name,
                    'teacher_name': t_name,
                    'room_name': r_name,
                    'is_shared': False,
                    'is_fixed': is_fixed
                }

    # 结果导向合班判定
    for d in range(5):
        for p in range(5):
            sig_counts = {}
            for cid in target_class_ids:
                item = scheds[cid][d][p]
                if item:
                    sig = f"{item['course']}_{item['teacher_name']}_{item['room_name']}"
                    sig_counts[sig] = sig_counts.get(sig, 0) + 1

            for cid in target_class_ids:
                item = scheds[cid][d][p]
                if item:
                    sig = f"{item['course']}_{item['teacher_name']}_{item['room_name']}"
                    if sig_counts[sig] >= 2:
                        item['is_shared'] = True

    # ==========================================
    # 绘图渲染区
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    days_str = ['周一', '周二', '周三', '周四', '周五']
    periods_str = ['第1-2节', '第3-4节', '第5-6节', '第7-8节', '第9-10节']

    for idx, cid in enumerate(target_class_ids):
        ax = axes[idx]
        name = target_class_names[idx]
        sched = scheds[cid]

        ax.axis('off')
        cell_text = [['' for _ in range(5)] for _ in range(5)]
        cell_colors = [[COLOR_EMPTY_BG for _ in range(5)] for _ in range(5)]

        for d in range(5):
            for p in range(5):
                item = sched[d][p]
                if item:
                    c_name_formatted = format_course_name(item['course'])
                    prefix = "[固]" if item['is_fixed'] else ""
                    cell_text[p][d] = f"{prefix}《{c_name_formatted}》\n{item['teacher_name']}\n{item['room_name']}"

                    if item['is_fixed']:
                        cell_colors[p][d] = COLOR_FIXED_BG
                    elif item['is_shared']:
                        cell_colors[p][d] = COLOR_SHARED_BG
                    else:
                        cell_colors[p][d] = COLOR_NORMAL_BG

        table = ax.table(cellText=cell_text, rowLabels=periods_str, colLabels=days_str,
                         cellColours=cell_colors, loc='center', cellLoc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        table.scale(1, 4.3)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(COLOR_GRID_LINE)
            cell.set_linewidth(0.8)
            cell.set_text_props(color=COLOR_TEXT_MAIN)

            if row == 0 or col == -1:
                cell.set_facecolor(COLOR_HEADER_BG)
                cell.set_text_props(weight='bold', fontsize=12)
            elif col >= 0 and row > 0 and sched[col][row - 1]:
                item = sched[col][row - 1]
                if item['is_shared'] or item['is_fixed']:
                    cell.set_text_props(weight='bold', color='#17202A')
            elif col >= 0 and row > 0 and not sched[col][row - 1]:
                cell.set_edgecolor('#E5E7E9')

        ax.set_title(f"行政班：{name}", fontsize=15, weight='bold', color=COLOR_TEXT_MAIN, pad=18)

    # ==========================================
    # 自定义颜色图例 (Legend Icons) - 已移除大标题
    # ==========================================
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_FIXED_BG, edgecolor='#AAB7B8', linewidth=1, label='固定预排课程 (硬约束优先)'),
        mpatches.Patch(facecolor=COLOR_SHARED_BG, edgecolor='#AAB7B8', linewidth=1, label='合班共享课程 (物理时空同步)'),
        mpatches.Patch(facecolor=COLOR_NORMAL_BG, edgecolor='#AAB7B8', linewidth=1, label='普通独立课程 (自主寻优)')
    ]

    # 将图标横向排开，放置在最上方
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=3, fontsize=13, frameon=False, columnspacing=3.0)

    # 释放上下呼吸空间，避免顶部太空
    plt.subplots_adjust(wspace=0.15, hspace=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 双格式高清导出
    png_path = os.path.join(output_dir, 'Fig1_Result_Oriented_4Classes_Final.png')
    svg_path = os.path.join(output_dir, 'Fig1_Result_Oriented_4Classes_Final.svg')

    plt.savefig(png_path, dpi=600, format='png')
    plt.savefig(svg_path, format='svg')
    plt.close()

    print(f"✨ 图1 (无大标题版) 绘制完美收工！")
    print(f"📄 位图保存至: {png_path}")
    print(f"📄 矢量图保存至: {svg_path}")


def main():
    parser = argparse.ArgumentParser(description="绘制四个代表性行政班课表")
    parser.add_argument("best_schedules", help="best_schedules.json 路径")
    parser.add_argument("--output-dir", default="paper_plots", help="图片输出目录")
    args = parser.parse_args()
    path_ablation = args.best_schedules
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(path_ablation):
        print(f"❌ 找不到 JSON 文件: {path_ablation}")
        return

    with open(path_ablation, 'r', encoding='utf-8') as f:
        schedules = json.load(f)

    try:
        print("🔗 正在连接数据库拉取最新实例数据...")
        loader = DataLoader('L1_Large', DB_CONFIG)
        data = loader.load()

        algo_key = next((k for k in schedules if "TrueCDM" in k), None)
        if algo_key is None and "Ours" in schedules:
            algo_key = "Ours"
        if algo_key:
            plot_4_classes_schedule(schedules[algo_key]['schedule'], data, output_dir)
            print(f"\n🎉 完美！双格式图表已成功导出至目录:\n📁 {output_dir}")
        else:
            print("⚠️ 未找到包含 TrueCDM 的算法数据。")
    except Exception as e:
        print(f"⚠️ 绘图失败: {e}")


if __name__ == "__main__":
    main()
