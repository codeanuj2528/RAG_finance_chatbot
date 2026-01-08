"""
budgeting.py - Budgeting Tools for Personal Finance
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def budgeting_tool():
    st.header("🛠️ Budgeting Tool")

    # Monthly Income Input
    income = st.number_input(
        "💵 Monthly Income",
        min_value=0.0,
        step=100.0,
        help="Your total monthly income after taxes."
    )

    # Expense Categories
    st.subheader("📊 Enter Your Monthly Expenses")
    
    expense_categories = {
        'Housing (Rent/Mortgage)': 0.0,
        'Utilities': 0.0,
        'Food': 0.0,
        'Transportation': 0.0,
        'Entertainment': 0.0,
        'Healthcare': 0.0,
        'Insurance': 0.0,
        'Debt Payments': 0.0,
        'Education': 0.0,
        'Savings & Investments': 0.0,
        'Miscellaneous': 0.0,
    }

    total_expenses = 0.0
    
    cols = st.columns(2)
    for i, category in enumerate(expense_categories):
        with cols[i % 2]:
            expense = st.number_input(
                f"{category}",
                min_value=0.0,
                step=10.0,
                key=f"expense_{category}"
            )
            expense_categories[category] = expense
            total_expenses += expense

    # Calculate Savings
    savings = income - total_expenses

    st.markdown("---")
    
    # Display Results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Income", f"${income:,.2f}")
    with col2:
        st.metric("Total Expenses", f"${total_expenses:,.2f}")
    with col3:
        if savings >= 0:
            st.metric("Monthly Savings", f"${savings:,.2f}", delta=f"{(savings/income*100):.1f}%" if income > 0 else None)
        else:
            st.metric("Monthly Deficit", f"${-savings:,.2f}", delta=f"-{(-savings/income*100):.1f}%" if income > 0 else None, delta_color="inverse")

    # Expense Breakdown Chart
    if income > 0 and total_expenses > 0:
        st.subheader("📈 Expenses Breakdown")
        
        # Filter non-zero expenses
        non_zero = {k: v for k, v in expense_categories.items() if v > 0}
        
        if non_zero:
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            wedges, texts, autotexts = ax1.pie(
                non_zero.values(),
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 9},
                pctdistance=0.8
            )
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            fig1.gca().add_artist(centre_circle)
            ax1.legend(
                wedges,
                non_zero.keys(),
                title="Categories",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=8
            )
            ax1.axis('equal')
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()

    # Budget Recommendations (50/30/20 Rule)
    if income > 0:
        st.subheader("💡 50/30/20 Budget Recommendations")
        
        needs = income * 0.5
        wants = income * 0.3
        savings_rec = income * 0.2

        col1, col2, col3 = st.columns(3)
        col1.metric("Needs (50%)", f"${needs:,.2f}")
        col2.metric("Wants (30%)", f"${wants:,.2f}")
        col3.metric("Savings (20%)", f"${savings_rec:,.2f}")

        # Calculate actual spending by category
        total_needs = sum([expense_categories[cat] for cat in 
            ['Housing (Rent/Mortgage)', 'Utilities', 'Food', 'Transportation', 'Healthcare', 'Insurance', 'Debt Payments']])
        total_wants = sum([expense_categories[cat] for cat in 
            ['Entertainment', 'Education', 'Miscellaneous']])
        total_savings = expense_categories.get('Savings & Investments', 0)

        # Comparison Chart
        st.subheader("📊 Recommended vs Actual")
        
        labels = ['Needs', 'Wants', 'Savings']
        recommended = [needs, wants, savings_rec]
        actual = [total_needs, total_wants, total_savings]

        x = np.arange(len(labels))
        width = 0.35

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        rects1 = ax2.bar(x - width/2, recommended, width, label='Recommended', color='#4CAF50')
        rects2 = ax2.bar(x + width/2, actual, width, label='Actual', color='#FF6B6B')

        ax2.set_ylabel('Amount ($)')
        ax2.set_title('Recommended vs. Actual Spending')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax2.annotate(
                    f'${height:,.0f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8
                )

        autolabel(rects1)
        autolabel(rects2)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Savings Goal Calculator
    st.subheader("🎯 Savings Goal Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        goal_amount = st.number_input(
            "Target Amount ($)",
            min_value=0.0,
            step=100.0
        )
    with col2:
        goal_timeframe = st.number_input(
            "Timeframe (months)",
            min_value=1,
            step=1,
            value=12
        )

    if goal_amount > 0 and goal_timeframe > 0:
        monthly_needed = goal_amount / goal_timeframe
        
        if savings >= monthly_needed:
            st.success(f"✅ You're on track! You need ${monthly_needed:,.2f}/month. You save ${savings:,.2f}/month.")
            months_to_goal = int(goal_amount / savings) if savings > 0 else 0
            st.info(f"At your current rate, you'll reach your goal in **{months_to_goal} months**.")
        elif savings > 0:
            st.warning(f"⚠️ You need ${monthly_needed:,.2f}/month but only save ${savings:,.2f}/month.")
            st.write(f"Gap: ${monthly_needed - savings:,.2f}/month")
        else:
            st.error("❌ Your current budget doesn't allow for savings toward this goal.")
