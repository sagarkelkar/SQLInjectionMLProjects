import streamlit as st
import utils.SQLiteDB as dbHandler
import utils.preprocessing


def main():
    st.title("SQL Injection Detection using ML")

    query = st.text_area('SQL Query Input', 'Enter your SQL query')

    if st.button("Generate"):
        pred = utils.preprocessing.getPrediction(query)

        if pred[0] == 0:
            print("It seems to be safe input")
            dbresponse = dbHandler.executeQuery(query)
            print("DB Response =", dbresponse)
            st.subheader("Output Result")
            st.markdown("""**:blue[It seems to be safe input]**""")
        else:
            st.subheader("Output Result")
            st.markdown("""**:red[ALERT :::: This can be SQL injection]**""")
            print("ALERT :::: This can be SQL injection")


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
