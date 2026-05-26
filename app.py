import streamlit as st
from langchain_core.messages import HumanMessage
from agents import build_graph

"""

"""

st.set_page_config(
    page_title = "AI Research Agent",
    layout = "wide"
)

st.title("AI Research Agent") # the heading at the top of the page

# --------------Sidebar------------------

with st.sidebar:
    st.header('configuration')

    thread_id = st.text_input(
        "Session ID",
        value="session-1",
        help="Same ID = Shared memory across conversation"
        # Text box for the session ID; same ID lets the agent remember prior turns
    )

    if st.button("clear session"):
        # If the user clicks the button, wipe the chat history
        st.session_state.message = [] # reset message list to empty
        st.success("Session cleared") # show a green confirmation banner

    st.divider() # draw a separator line

    st.header('How it works?')
    st.markdown("""
    1. Enter a research question
    2. Agent decomposes it into sub-queries
    3. Search arXiv 
    4. Downloads papers
    5. Generates report
    """)
    # Render an explanation of the agent's workflow

# ------------Session state init-----------------

# create an empty list to store chat on the first load
if "messages" not in st.session_state:
    st.session_state.messages = []

#----------------- Render chat history--------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # render each stored message in a chat bubble
        st.markdown(message["content"])

# ------------Handle new user input-------------------

if prompt := st.chat_input("Enter your research question:"):
    # show the input box at the bottom
    # thd block below only runs when user submits a message

    # display user message and save the new message
    st.session_state.messages.append({"role":"user","content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt) # render user-style bubble

    # ---------- Agent execution ---------------
    with st.chat_message("assistant"):

        # live update agent's status
        status_placeholder = st.empty()
        report_placeholder = st.empty()

        try:
            graph = build_graph()

            config = {"configurable":{"thread_id":thread_id}}

            with st.spinner("🔍 Researching..."):
                # show a loading spinner while agent runs

                # streaming: yield a chunk as each node finished
                for chunk in graph.stream(
                    {"messages":[HumanMessage(content=prompt)]},
                    config = config,
                    stream_mode = "updates" # Yield output after each node complete
                ):

                    # iterate over each node's output when it finishes
                    # if output include current_step, the status line will be updated. user will see sm like: Searching arXiv...
                    for node_name, node_output in chunk.items():
                        current_step = node_output.get("current_step","")
                        if current_step:
                            status_placeholder.info(
                                f"**{node_name}**:{current_step}"
                            )
                            # show live time progress

            # ----------------Retrieve final result -------------------

            state = graph.get_state(config)
            final_report = state.values.get("final_report", "")

            status_placeholder.empty()  # Remove the last status message

            if final_report:
                report_placeholder.markdown(final_report) # render report

                # save the report to chat history
                st.session_state.messages.append({
                    "role":"assistant",
                    "content":final_report
                })

            else:
                report_placeholder.warning(
                    "No report generated. Try a more specfic questiion."
                )

        except Exception as e:
            st.error(f"Error:{str(e)}") # show a concise red error banner
            st.exception(e)
            # print the full traceback



