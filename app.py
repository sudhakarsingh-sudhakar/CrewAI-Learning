from crewai import Agent, tools,LLM,Crew,Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the CrewAI agent
topic = "Reliance Industries Stock Price analysis"
#Tool - 1

api_key = os.getenv("GOOGLE_API_KEY")
llm = LLM(model='gemini/gemini-1.5-flash',api_key=api_key)

#llm = LLM(model="gpt-4")
#Tool - 2
search_tool = SerperDevTool(n=10)

#Agent 1
senior_research_analyst = Agent(
    role="Senior Research Analyst",
    goal=f"Research, Analysis and syntehsize comprehensive infomation of {topic} from multiple sources.",
    backstory="Research Analyst with expertise in stock market analysis and financial data interpretation.",
    allow_delegation=False,
    verbose=True,
    tools=[search_tool],
    llm=llm
)

#Agent 2 content writer
content_writer = Agent(
    role="Content Writer",
    goal=f"Transform research finding into a well-structured report while ensuring clarity and coherence and maintaining the original intent of the information.",
    backstory="You are a content writer with expertise in financial writing and report generation.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

#Research Task  

research_task = Task(
    description=f"""conduct a comprehensive research on {topic} and provide a detailed analysis.:
    1. Gather information from multiple sources.
    2. Analyze the data and identify key trends, patterns, and insights.
    3. Summarize the findings in a clear and concise manner.
    4. Provide recommendations based on the analysis.
    5. Ensure the report is well-structured and easy to understand.
    6. Include relevant data, charts, and graphs to support the analysis.
    7. Ensure the report is free from plagiarism and properly cited.
    8. Ensure the report is well-formatted and visually appealing.
    9. Ensure the report is tailored to the target audience.
    10. Ensure the report is delivered on time.""",
    expected_output="Research findings and analysis report",
    agent=senior_research_analyst,

)

#Content Writing Task
content_writing_task = Task(
    description=("""Transform the research findings into a well-structured report while ensuring clarity and coherence and maintaining the original intent of the information.:
    
                 attention to detail, ensuring that the report is free from errors and inconsistencies.
                 1. Review the research findings and analysis report.
                 2. Identify the key points and insights that need to be included in the report.
                 3. Organize the information in a logical and coherent manner.
                 4. Write the report in a clear and concise manner, ensuring that the language is appropriate for the target audience.
                 5. Ensure that the report is well-structured, with clear headings and subheadings."""),
    expected_output="Final report",
    agent=content_writer,
)


# Create a Crew instance

crew = Crew(
    agents=[senior_research_analyst, content_writer],
    tasks=[research_task, content_writing_task],
    verbose=True,
)

# Run the Crew
results = crew.kickoff(inputs={"topic": topic})
# Print the results
print("Research Findings and Analysis Report:")
print(results)
