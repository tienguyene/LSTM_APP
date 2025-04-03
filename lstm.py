import streamlit as st
from vnstock import Vnstock
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import joblib
import function as ut
import base64
from streamlit_echarts import st_echarts
st.set_page_config(layout="wide")

def get_base64(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64("background.png")


st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }} /* ảnh nền */

    header {{
        background-color: #03001C !important;
    }} /* màu nền header */

    [data-testid="stSidebar"] {{
        background-color: #03001C !important;
    }} /* màu nền sidebar */

    [data-testid="stSidebar"] h1 {{
        font-size: 28px !important;
        background-color: #232D3F !important;
        color: #FFFFFF !important;
        text-align: center !important;
        padding: 10px;
        border-radius: 8px;
        border: 2px solid #301E67;
    }} /* tiêu đề sidebar */

    .stRadio [data-testid="stWidgetLabel"] [data-testid="stMarkdownContainer"] {{
        font-size: 25px !important;
        padding: 10px 0px 10px 110px;
    }} /* tiêu đề radio button - Pages */

    [role="radiogroup"] {{
        background-color: #232D3F !important;
        padding: 10px;
        border-radius: 8px;
        border: 2px solid #301E67;
    }} /* màu nền radio */

    [data-baseweb="radio"] [data-testid="stMarkdownContainer"]{{
        font-size: 20px !important;
        color: #FFFFFF !important;
    }}  /* chữ radio */

    div.st-au {{
        margin: 10px 0px 0px 0px !important;
    }} /* khoảng cách giữa các nút radio */

    div.home_page {{
        background-color: #232D3F !important;
        opacity: 0.9 !important;
        padding: 0px 0px 0px 20px;
        
        border-radius: 10px;
        border: 2px solid #301E67;
        min-height: 600px;
    }} /* màu nền trang chính */

    div.home_page h1 {{
        font-size: 45px !important;
        color: #FFFFFF !important;
        text-align: center !important;
    }} /* tiêu đề trang chính */

    div.home_page #short_des {{
        font-size: 20px !important;
        color: #FFFFFF !important;
        text-align: center !important;
        margin-bottom: 20px;
    }} /* mô tả ngắn */

    div.home_page h2 {{
        font-size: 28px;
        color: #FFFFFF !important;
        margin-left: 40px;
        margin-top: 50px;
    }} /* tiêu đề phần 2 */

    div.home_page #keys {{
        font-size: 18px;
        margin-left: 60px;
        color: #FFFFFF !important;
        margin-bottom: 10px;
    }} /* tiêu đề phần 3 */

    div.home_page ul {{
        padding-left: 50px;
        font-size: 18px;
        color: #FFFFFF !important;
    }} /* danh sách các tính năng */

    div.home_page li {{
        margin-bottom: 10px;
        color: #FFFFFF !important;
    }} /* khoảng cách giữa các mục trong danh sách */

    .stSelectbox [data-testid="stMarkdownContainer"] {{
        font-size: 25px !important;
        color: #FFFFFF !important;
    }} /* tiêu đề các select box */

    .st-cq {{
        height: 2.5rem;
    }} /* chiều cao của select box */

    [data-testid="stSidebar"] [data-testid="stButton"] [data-testid="stBaseButton-secondary"] {{
        width: 100%;
        background-color: #232D3F !important;
        color: #FFFFFF !important;
        border-radius: 8px;
        border: 2px solid #301E67;
        font-size: 20px !important;
    }}

    .stMain [data-testid="stSelectbox"] {{
        background-color: #232D3F !important;
        border-radius: 8px;
        border: 2px solid #301E67;
        padding: 10px;
        opacity: 1 !important;
    }}

    .stMain [data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] {{
        font-size: 20px !important;
        color: #FFFFFF !important;

    }} /* chữ trong select box */

    li[role="option"] {{
        color: white !important;
        padding: 10px;
        border-radius: 5px;
        width: 100%;
        margin: 5px 0px;
    }}

    [data-testid="stSelectboxVirtualDropdown"] {{
        background-color: #232D3F !important;
        border-radius: 8px;
        border: 2px solid #301E67;
    }}

    [data-testid="stBaseButton-secondary"]  {{
        width: 100%;
    }}

    #choose-a-stock-to-predict {{
        font-size: 30px !important;
        color: #FFFFFF !important;
        text-align: right !important;
    }}

    #stock-trend-plot {{
        font-size: 30px !important;
        color: #FFFFFF !important;
        text-align: right !important;
    }}

    #company-information {{
        font-size: 30px !important
        color: #FFFFFF !important;
        text-align: right !important;
    }}

    .chart_desc {{
        font-size: 30px !important;
        color: #FFFFFF !important;
        text-align: right !important;
    }}

    [data-baseweb="progress-bar"] > div > div > div {{
        background-color: #FFD700 !important;  /* Chỉnh màu của thanh tiến trình */
    }}

    .stAlert p {{
        font-size: 25px !important;
        color: #FFFFFF !important;
        text-align: right !important;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)

def load_data():
    if "stock" not in st.session_state:
        st.session_state.stock = Vnstock().stock(symbol='ACB', source='VCI')
        print("Stock loaded")
    if "Vn100_list" not in st.session_state:
        st.session_state.Vn100_list = st.session_state.stock.listing.symbols_by_group('VN100')
        print("Vn100_list loaded")
    if "stock_dict" not in st.session_state:
        st.session_state.stock_dict, date_list = ut.read_data('stock_data_latest.csv', st.session_state.Vn100_list)
        st.session_state.formated_date_list = [i.date() for i in date_list]
        print("Stock dict loaded")
    if "models" not in st.session_state:
        tmp = [1, 3, 3, 2, 1, 2, 2, 2, 3, 2]
        st.session_state.models = [load_model(f"MODELS/model_{i+1}_{id}.keras") for i, id in enumerate(tmp)]
        print("Models loaded")
    if "predictions" not in st.session_state:
        st.session_state.predictions = None
        print("Predictions loaded")
    if "index_show" not in st.session_state:
        st.session_state.index_show = 1
        print("index_show loaded")
    if "index" not in st.session_state:
        st.session_state.index = None
        print("Index loaded")
    if "result_1" not in st.session_state:
        st.session_state.predictions_df = None
        print("result_1 loaded")

    if "result_2" not in st.session_state:
        st.session_state.result_2 = None
        print("result_2 loaded")

    if "company_info" not in st.session_state:
        st.session_state.company_info = pd.read_csv("company_info.csv")
        print("company_info loaded")

def take_x_dates(stock_dict, stock_id, index, num_dates):
    return stock_dict[stock_id].iloc[-num_dates-index-1:-index-1:1]

def predict_single_stock(model, input):
    scaler = MinMaxScaler()
    temp = scaler.fit_transform(input)

    if type(model) == Sequential:
        prediction = model.predict(np.expand_dims(temp, axis=0))
        prediction_reshaped = np.zeros((prediction.shape[0], 5)) # Tạo một mảng 2 chiều với 5 cột
        prediction_reshaped[:, 3] = prediction[:, 0].flatten() # Cột thứ 4 của mảng này sẽ chứa giá cổ phiếu dự đoán
        prediction = scaler.inverse_transform(prediction_reshaped)[:, 3] # Chuyển giá cổ phiếu dự đoán về dạng ban đầu (trước khi chuẩn hóa) để được giá trị thực   
    else:
        prediction = model.predict(temp.flatten().reshape(1, -1))
        prediction_reshaped = np.zeros((prediction.shape[0], 5)) # Tạo một mảng 2 chiều với 5 cột
        prediction_reshaped[:, 3] = prediction # Cột thứ 4 của mảng này sẽ chứa giá cổ phiếu dự đoán
        prediction = scaler.inverse_transform(prediction_reshaped)[:, 3]

    return prediction


def predict_all_stocks(stock_dict, index, num_dates, stock_list, model):
    predictions = []
    for stock_id in stock_list:
        pred = predict_single_stock(model, take_x_dates(stock_dict, stock_id, index, num_dates))
        predictions.append(pred)
    return np.array(predictions).reshape(1, -1)
    
def predict_with_all_models(stock_dict, index, num_dates, stock_list, models):
    n = len(models)
    predictions = np.empty((0, 100))
    progress_bar  = st.empty()
    for i in range(n):
        if i <= 9:
            predictions = np.concatenate((predictions, predict_all_stocks(stock_dict, index, num_dates[i], stock_list, models[i])), axis = 0)
        else:
            predictions = np.concatenate((predictions, predict_all_stocks(stock_dict, index, 30, stock_list, models[i])), axis = 0)
        progress_bar.progress((i+1)/n)
    st.warning("Complete!")
    return predictions

def compare_with_real_data(stock_dict, index, stock_list, predictions, show_comparison = 1, index_show = 1):
    actual = np.array([stock_dict[stock_id].iloc[-(index+1)][3] for stock_id in stock_list])
    temp = pd.DataFrame(predictions)
    cleaned = temp.iloc[:,:].reset_index(drop=True)
    pred_df = cleaned.T
    pred_df.columns = [f"LSTM_{i+1}" for i in range(10)]
    pred_df.insert(0, "Stock ID", stock_list)
    pred_df.insert(1, "Actual", actual)


    err_matrix = abs(predictions[:] - actual)
    mae = np.sum(err_matrix, axis = 1) / len(stock_list)
    mae = pd.DataFrame(mae)
    mae_df = mae.T
    mae_df.columns = [f"MAE_{i+1}" for i in range(mae_df.shape[1])]
    if show_comparison == 1:
        st.data_editor(pred_df.iloc[((index_show - 1) * 10):((index_show) * 10)])
        # st.data_editor(mae_df)

    return pred_df, mae_df  ##################

def predict_future_days(stock_dict, selected_stock, model_index, days_to_predict, training_days, Vn100_list, models):
    stock_idx = Vn100_list.index(selected_stock)
    model = models[model_index]
    stock_data = stock_dict[selected_stock].iloc[-(training_days+1):, 3].values
    predictions = []
    for i in range(days_to_predict):
        input_data = stock_data[-training_days:]
        input_data = input_data.reshape(1, -1)
        prediction = predict_single_stock(model, input_data)
        predictions.append(prediction[0])
        stock_data = np.append(stock_data, prediction[0])
    return predictions




def calculate_expected_return(predictions, stock_dict, index, stock_list):
    previous = np.array([stock_dict[stock_id].iloc[-(index+2)][3] for stock_id in stock_list])
    expected_return = (predictions - previous) / previous
    return expected_return

def select_top_stocks(expected_return, num_stocks=4):
    if expected_return.ndim == 1:
        top_indices = np.argsort(expected_return)[-num_stocks:][::-1]
    else:
        top_indices = np.argsort(expected_return, axis=1)[:, -num_stocks:][:, ::-1]
    return top_indices

def top_stocks_return(expected_return, top_indices):
    if expected_return.ndim == 1:
        return expected_return[top_indices]
    else:
        return np.take_along_axis(expected_return, top_indices, axis=1)
    
def covariance_matrix(stock_dict, stock_list, index, num_data=60):
    closing_prices = np.array([stock_dict[stock_id].iloc[-(num_data+index+2):-(index+1), 3].values for stock_id in stock_list]).T
    daily_returns = (closing_prices[1:] - closing_prices[:-1]) / closing_prices[:-1]
    cov_matrix = np.cov(daily_returns, rowvar=False)
    return cov_matrix


def monte_carlo_simulation(num_simulations, top_returns, stock_dict, top_stocks, index, num_data=60):
    cov_matrix = covariance_matrix(stock_dict, top_stocks, index, num_data)
    portfolios = []
    for _ in range(num_simulations):
        weights = np.random.dirichlet(np.ones(len(top_stocks)))
        port_return = np.dot(weights, top_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        portfolios.append((weights, port_return, port_volatility))
    portfolios = sorted(portfolios, key=lambda x: x[1] / x[2], reverse=True)
    return portfolios

def negative_sharpe(weights, top_returns, stock_dict, top_stocks, index, num_data=60, risk_free_rate=0):
    port_returns = np.dot(weights, top_returns)
    cov_matrix = covariance_matrix(stock_dict, top_stocks, index, num_data)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (port_returns - risk_free_rate) / port_volatility
    return -sharpe_ratio

def mean_variance_simulation(top_returns, stock_dict, top_stocks, index, num_data=60, risk_free_rate=0):
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = [(0.001, 0.999) for _ in top_stocks]
    initial_guess = np.ones(len(top_stocks)) / len(top_stocks)
    result = minimize(negative_sharpe, initial_guess, args=(top_returns, stock_dict, top_stocks, index, num_data, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    best_weights = result.x
    best_return = np.dot(best_weights, top_returns)
    cov_matrix = covariance_matrix(stock_dict, top_stocks, index, num_data)
    best_volatility = np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix, best_weights)))
    return best_weights, best_return, best_volatility
    
def get_pie_options(title, data):
    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "item"},
        "legend": {"orient": "vertical", "left": "left", "bottom": "0%"},
        "backgroundColor": "white",
        "series": [{
            "name": title,
            "type": "pie",
            "radius": ["40%", "70%"],  # Tạo Donut chart
            "avoidLabelOverlap": False,
            "label": {"show": True, "formatter": "{b}: {d}%"},
            "data": data
        }]
    }

def display_company_data(df, stock):
    stock_data = df[df['stock'] == stock]
    stock_data = stock_data.map(lambda x: x.lstrip() if isinstance(x, str) else x)


    if not stock_data.empty:
        stock_data_transposed = stock_data.T.reset_index()
        stock_data_transposed.columns = ["Attribute", "Value"]  # Đặt tên cột

        # Hiển thị DataFrame với cột Website có thể click
        st.data_editor(
            stock_data_transposed,
            hide_index=True,
            column_config={
                "Attribute": st.column_config.TextColumn(label="Attribute"),
                "Value": st.column_config.TextColumn(label="Content")
            }
        )
        # st.data_editor(
        #     stock_data,
        #     hide_index=True,
        #     column_config={
        #         "company_name": st.column_config.TextColumn(label="company_name"),
        #         "company_profile": st.column_config.TextColumn(label="company_profile"),
        #         "history_dev": st.column_config.TextColumn(label="history_dev"),
        #         "key_developments": st.column_config.TextColumn(label="key_developments"),
        #         "business_strategies": st.column_config.TextColumn(label="business_strategies"),
        #         "website": st.column_config.LinkColumn(label="website"),
        #     }
        # )
    else:
        st.error("No data found for the selected stock.")


    


def main():
    load_data() # Gọi hàm để load các dữ liệu và lưu trong session (không cần load lại khi chuyển trang)


    st.sidebar.markdown("# PAGES")

    page = st.sidebar.radio("", ["Home 🏠", "Stock Prediction 🔮", "Portfolio Optimization 🔢"])

    if page == "Home 🏠":
        st.markdown(
            """
            <div class="home_page">
                <h1>Stock Prediction & Portfolio Optimization 📈</h1>   
                <p id="short_des">An AI-powered application that helps you make smarter financial decisions.</p>    
                <h2>🚀 Welcome to <span style="color:#FFD700;">VN100 Stock Prediction & Portfolio Optimization</span></h2>            
                <p id="keys">✨ Key Features:</p>
                <ul>
                    <li>🔮 <strong>Stock Price Prediction</strong> - Predict the prices of VN100 stocks for a selected future date.</li>
                    <li>🔢 <strong>Portfolio Optimization</strong> - Construct an optimal investment portfolio using Monte Carlo simulations and Mean-Variance techniques.</li>
                    <li>📊 <strong>Visual Insights</strong> - Get useful charts, including a pie chart that shows asset allocation in your portfolio.</li>
                    <li>🚀 <strong>Start Exploring Now!</strong> - Navigate through the pages to make data-driven investment decisions.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif page == "Stock Prediction 🔮":
        # Đọc dữ liệu đã lưu trữ
        stock_dict = st.session_state.stock_dict
        formated_date_list = st.session_state.formated_date_list
        Vn100_list = st.session_state.Vn100_list

        if not isinstance(Vn100_list, list):
            Vn100_list = Vn100_list.tolist()


        # st.markdown(f"##### 1. Dữ liệu được lấy từ ngày {formated_date_list[0]} đến ngày {formated_date_list[-1]}")
        selected_day = st.selectbox(f"📅 Select a day", formated_date_list[-1:-16:-1])
        if selected_day is not None:
            sd_index = (formated_date_list[::-1]).index(selected_day)
            if st.session_state.index != sd_index:
                st.session_state.index = sd_index
                st.session_state.result_1 = None
                st.session_state.result_2 = None
                st.session_state.predictions = None

        if st.sidebar.button("Predict"):
            st.session_state.index_show = 1
            predictions = predict_with_all_models(stock_dict, sd_index, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], Vn100_list, st.session_state.models)
            pred_df, mae_df = compare_with_real_data(stock_dict, sd_index, Vn100_list, predictions, show_comparison=0, index_show=st.session_state.index_show)
            st.session_state.predictions = predictions
            st.session_state.result_1 = (pred_df, mae_df)

        col1, col4 = st.sidebar.columns([1, 1])
        with col1:
            if st.button("⬅️"):
                if st.session_state.index_show > 1:
                    st.session_state.index_show -= 1
                    st.rerun()
        with col4:
            if st.button("➡️"):
                if st.session_state.index_show < 10:
                    st.session_state.index_show += 1
                    st.rerun()

        if st.session_state.predictions is not None:
            compare_with_real_data(stock_dict, st.session_state.index, Vn100_list, st.session_state.predictions, show_comparison=1, index_show=st.session_state.index_show)

        # Thêm bảng 10x10 và biểu đồ
        
        if st.session_state.predictions is not None:
            model_choice = st.selectbox("Choose model to predict stock trend", [f"LSTM {i+1}" for i in range(10)])
            
            if model_choice.startswith("LSTM"):
                model_index = int(model_choice.split()[1]) - 1

            # Tạo bảng 10x10 với các nút Streamlit thay vì HTML
            st.markdown(
                """
                <h3> Choose a stock to predict </h3>
                """,
                unsafe_allow_html=True
            )
            table_data = [Vn100_list[i:i+10] for i in range(0, 100, 10)]

            for row in table_data:
                cols = st.columns(10)
                for i, stock in enumerate(row):
                    if cols[i].button(stock, key=stock):
                        st.session_state.selected_stock = stock
                        st.toast("### ⬇ Scroll down to see the graph!")

            # Khởi tạo selected_stock trong session_state nếu chưa có
            if "selected_stock" not in st.session_state:
                st.session_state.selected_stock = None

            selected_stock = st.session_state.selected_stock

            # Hiển thị biểu đồ nếu có mã cổ phiếu được chọn
            if selected_stock and st.session_state.predictions is not None:
                # Nếu có mã cổ phiếu được chọn, hiển thị thông tin công ty
                if selected_stock:
                    st.markdown(
                        """
                        <h3> Company Information </h3>
                        """,
                        unsafe_allow_html=True
                    )
                    display_company_data(st.session_state.company_info, selected_stock)
                
                stock_df = stock_dict[selected_stock]
                if stock_df.empty or 'close' not in stock_df.columns:
                    st.error("Can not find data for the selected stock.")
                    return
                else:
                    try:
                        st.markdown(
                            """
                            <h3> Stock trend plot </h3>
                            """,
                            unsafe_allow_html=True
                        )
                        # Lấy dữ liệu 10 ngày trước
                        stock_data = stock_df.iloc[-(st.session_state.index+11):-(st.session_state.index+1)]['close'].values
                        dates = formated_date_list[-(st.session_state.index+11):-(st.session_state.index+1)]

                        # Lấy giá trị dự đoán
                        stock_idx = Vn100_list.index(selected_stock)
                        predicted_price = st.session_state.predictions[model_index][stock_idx]
                        
                        # Tạo dữ liệu cho biểu đồ
                        extended_dates = list(dates) + [formated_date_list[-(st.session_state.index+1)]]
                        extended_prices = list(stock_data) + [predicted_price]
                        
                        
                        # Vẽ biểu đồ
                        fig, ax = plt.subplots(figsize=(12, 6))
                        # Truyền extended_dates làm tọa độ x
                        ax.plot(extended_dates, extended_prices, label="Actual price")
                        # Đường dự đoán riêng để nổi bật
                        ax.plot(extended_dates[-2:], extended_prices[-2:], linestyle='-', color='red', label="Predicted price")
                        
                        # Định dạng biểu đồ
                        ax.set_title(f"Price {selected_stock}")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price (VND)")
                        ax.legend()
                        
                        # Đặt ticks và nhãn trục x
                        
                        ax.set_xticks([])
                        ax.set_xticklabels([])
                        # ax.get_xaxis().set_visible(False)
                        # ax.grid(True)
                        fig.tight_layout()
                        
                        # Hiển thị biểu đồ trong Streamlit
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error: {e}")

    elif page == "Portfolio Optimization 🔢":
        stock_dict = st.session_state.stock_dict
        Vn100_list = st.session_state.Vn100_list
        predictions = st.session_state.predictions
        index = st.session_state.index

        model_choice = st.selectbox("Choose model to construct portfolios", [f"LSTM {i+1}" for i in range(10)])

        if model_choice.startswith("LSTM"):
            model_index = int(model_choice.split()[1]) - 1

        if st.sidebar.button("Construct portfolio"):
            selected_prediction = predictions[model_index]

            print(model_index)
            
            expected_return = calculate_expected_return(selected_prediction, stock_dict, index, Vn100_list)
            top_indices = select_top_stocks(expected_return, num_stocks=4)
            top_stocks = [Vn100_list[i] for i in top_indices]
            top_returns = expected_return[top_indices]
            
            # Monte Carlo
            portfolios = monte_carlo_simulation(10000, top_returns, stock_dict, top_stocks, index)
            best_portfolio = portfolios[0]
            # st.write(f"Best portfolio từ Monte Carlo: weights={best_portfolio[0]}, return={best_portfolio[1]}, volatility={best_portfolio[2]}")

            
            # Mean-Variance
            best_weights, best_return, best_volatility = mean_variance_simulation(top_returns, stock_dict, top_stocks, index)
            # st.write(f"Optimal portfolio: weights={best_weights}, return={best_return}, volatility={best_volatility}")

            top_stocks_df = pd.DataFrame(top_stocks)
            top_stocks_df.columns = ["Stock ID"]
            top_stocks_df.insert(1, "Expected Return", top_returns)
            # top_stocks_df.insert(2, "MC_weights", best_portfolio[0])
            top_stocks_df.insert(2, "MV_weights", best_weights)

            new_rows = pd.DataFrame([
                ["Returns", None, best_return],
                ["Volatility", None, best_volatility]],
                columns=["Stock ID", "Expected Return", "MV_weights"]
            )

            top_stocks_df = pd.concat([top_stocks_df, new_rows], ignore_index=True)

            if st.session_state.result_2 is not None:
                if model_index not in [x - 1 for x in list(st.session_state.result_2.keys())]:
                    st.session_state.result_2[int(model_index)+1] = top_stocks_df
            else:
                st.session_state.result_2 = {int(model_index)+1: top_stocks_df}

            for i in list(st.session_state.result_2.keys())[::-1]:
                name = ""
                if i >= 0 and i <= 10:
                    name = f"LSTM_{i}"

                st.markdown(f'<div class="chart_desc"> Portfolio {name} </div>', unsafe_allow_html=True)
                # st.dataframe(st.session_state.result_2[i])


                # mc_weights = st.session_state.result_2[i]["MC_weights"][:4]
                mv_weights = st.session_state.result_2[i]["MV_weights"][:4]
                top_stocks_3 = st.session_state.result_2[i]["Stock ID"][:4]
    
                # mc_data = [{"name": top_stock_3, "value": mc_weight} for top_stock_3, mc_weight in zip(top_stocks_3, mc_weights)]
                mv_data = [{"name": top_stock_3, "value": mv_weight} for top_stock_3, mv_weight in zip(top_stocks_3, mv_weights)]

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.data_editor(st.session_state.result_2[i])
                with col2:
                    st_echarts(options=get_pie_options("Mean-Variance Optimization", mv_data), height="400px")
if __name__ == '__main__':
    main()
