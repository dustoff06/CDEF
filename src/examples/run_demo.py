from cdef_analyzer.core import run_demo_scenarios

cat > run_demo.py << 'EOF'
#!/usr/bin/env python3
"""Run CDEF demonstration scenarios."""

from cdef_analyzer import run_demo_scenarios


def main():
    """Run the 4-scenario CDEF demonstration"""
    df_results, _ = run_demo_scenarios()
    print("\n✓ Demo complete!")
    return df_results


if __name__ == "__main__":
    results = main()
EOF

python run_demo.py
