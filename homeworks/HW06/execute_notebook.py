#!/usr/bin/env python3
"""
Скрипт для выполнения всех ячеек Jupyter notebook и создания артефактов.
"""
import json
import sys
import traceback
from pathlib import Path

def execute_notebook(notebook_path):
    """Выполняет все code ячейки из notebook последовательно."""
    notebook_path = Path(notebook_path)
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Глобальное пространство имен для выполнения ячеек
    global_namespace = {}
    
    # Фильтруем только code ячейки
    code_cells = [(i, cell) for i, cell in enumerate(notebook['cells']) 
                  if cell.get('cell_type') == 'code']
    
    total = len(code_cells)
    print(f"Найдено code ячеек: {total}\n")
    print("=" * 80)
    
    errors = []
    
    for idx, (cell_idx, cell) in enumerate(code_cells, 1):
        source = ''.join(cell.get('source', []))
        
        if not source.strip():
            continue
        
        print(f"\n[{idx}/{total}] Выполнение ячейки {cell_idx}...")
        print("-" * 80)
        
        # Показываем первые 2-3 строки кода
        lines = [line.strip() for line in source.split('\n') if line.strip()][:3]
        for line in lines:
            if len(line) > 100:
                print(f"  {line[:97]}...")
            else:
                print(f"  {line}")
        
        try:
            exec(source, global_namespace)
            print("✓ Успешно")
        except KeyboardInterrupt:
            print("\n✗ Прервано пользователем")
            errors.append(f"Ячейка {cell_idx}: KeyboardInterrupt")
            break
        except Exception as e:
            error_msg = f"Ячейка {cell_idx}: {type(e).__name__}: {str(e)}"
            errors.append(error_msg)
            print(f"✗ {error_msg}")
            # Для некоторых ошибок продолжаем (например, если переменные не определены из-за незавершенных предыдущих ячеек)
            if isinstance(e, (NameError, AttributeError)) and ('best_rf' in str(e) or 'best_gb' in str(e) or 'y_proba_best' in str(e)):
                print("  (Пропускаем - переменные будут определены позже)")
                continue
    
    print("\n" + "=" * 80)
    if errors:
        print(f"\nНайдено ошибок: {len(errors)}")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✓ Все ячейки выполнены успешно!")
        return True

if __name__ == '__main__':
    notebook_path = Path('HW06.ipynb')
    if not notebook_path.exists():
        print(f"Ошибка: файл {notebook_path} не найден!")
        sys.exit(1)
    
    print("ВНИМАНИЕ: Выполнение может занять много времени из-за GridSearchCV!")
    print("RandomForest: ~540 фитов, GradientBoosting: ~405 фитов")
    print("Начинаю выполнение...\n")
    
    success = execute_notebook(notebook_path)
    sys.exit(0 if success else 1)

