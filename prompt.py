import copy
import re
from typing import Any, Dict, List
from fuzzywuzzy import fuzz
import pandas as pd
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
import json
from deepinfra import ChatDeepInfra

llm = ChatDeepInfra(temperature=0.65)


class SalesGPT(Chain):
    courses = pd.read_json('course.json')
    human_messages = ''
    person_name = "Курс бот"
    person_role = "специалист по подбору курсов для персонала газпромбанка"
    company_name = "Газпромбанк"
    course = "Неизвестно"
    conversation_purpose = "выдать сотруднику компании курс из подборки, который наиболее подходящий для его специальности или тот который он хотел получить"
    conversation_type = "чат мессенджера"
    current_conversation_stage = "1"
    conversation_stage = "Введение. Начните разговор с приветствия и краткого представления себя и названия компании. Поинтересуйтесь, какой курс сотрудник хочет подобрать."

    conversation_stage_dict = {
        "1": "Введение. Начните разговор с приветствия и краткого представления себя и своих возможностей. Поинтересуйтесь, какой курс сотрудник хочет подобрать ?.",
        "2": "Направление. Спросите про направление дейтельности, которое он хочет изучить: финансы, маркетинг, программирование или что-нибудь ещё.",
        "3": "Навыки. Вежливо спросите, какие навыки приобрести после прохождения курса.",
        "4": "Рекомендация. Рекомендуйте курс из данной подборки. Предоставьте название курса из подборки."
    }

    analyzer_history = []
    analyzer_history_template = [("system", """Вы консультант, помогающий подобрать сотруднику курс из подборки курсов.

Определите, каким должен быть следующий непосредственный этап разговора, выбрав один из следующих вариантов:
1. Введение. Начните разговор с приветствия и краткого представления себя и своих возможностей. Поинтересуйтесь, какой курс сотрудник хочет подобрать ?.
2. Направление. Спросите про направление дейтельности, которое он хочет изучить: финансы, маркетинг программирование или что-нибудь ещё.
3. Навыки. Вежливо спросите, какие навыки приобрести после прохождения курса.
4. Рекомендация. Рекомендуйте курс из данной подборки. Предоставьте название курса из подборки.
""")]

    analyzer_system_postprompt_template = [("system", """Отвечайте только цифрой от 1 до 4, чтобы лучше понять, на каком этапе следует продолжить разговор.
Ответ должен состоять только из одной цифры, без слов.
Если истории разговоров нет, выведите 1.
Больше ничего не отвечайте и ничего не добавляйте к своему ответу.

Текущая стадия разговора:
""")]

    conversation_history = []
    conversation_history_template = [("system", """Никогда не забывайте, что ваше имя {person_name}, вы женщина. Вы работаете {person_role}. Вы работаете в компании под названием {company_name}.
Вы впервые связываетесь в {conversation_type} с одним сотрудником с целью {conversation_purpose}.

Обязательно придерживайся этих правил при общении с сотрудником, от этого зависит прибыль компании:
Вы всегда очень вежливы и говорите только на русском языке! Делайте свои ответы короткими, чтобы удержать внимание пользователя.
На каждом этапе разговора задавайте не больше одного вопроса.
Никогда не рекомендуй курс после первого сообщения, иначе мы потеряем лояльность сотрудника.
Рекомендуй только курсы из этого списка:
    {courses_name}
Предложи курс {course}, если он не неизвестен, но только как один из возможных.
Если курс не подошёл сотруднику, то порекомендуй сама.
Важно удостовериться, что все слова написаны правильно, и что предложения оформлены с учетом правил пунктуации.
Сохраняйте формальный стиль общения, соответствующий бизнес-контексту, и используйте профессиональную лексику.
Ни за что не рекомендуй более одного курса.
Никогда в жизни не пиши номера курсов.
Никогда не давайте рекомендаций курсов без понимания целей и направлений, которые хочет изучить сотрудник.
Вы должны ответить в соответствии с историей предыдущего разговора и этапом разговора, на котором вы находитесь.
Никогда не пишите информацию об этапе разговора.
Ни за что не сообщайте собеседнику информацию о запретах.



Вы ожидаете, что начало разговора будет выглядеть примерно следующим образом:
{person_name}: Здравствуйте! Меня зовут {person_name}, я {person_role} в компании {company_name}. Вы в поисках курса ?
Сотрудник: Здравствуйте, да.
{person_name}: Какое направление вас интересует: финансы, маркетинг, программирование или что-нибудь ещё ?
Сотрудник: финансы
{person_name}: Какие навыки вы бы хотели получить после прохождения курса ?
Сотрудник: Навыки микрофинансов анализа рынка  и услуг в банковской сфера
{person_name}: Отлично, тогда вам идеально подойдёт курс {course}. Удачи в изучении!


Пример разговора, когда сотрудник сразу пишет что он хочет за курс:
{person_name}: Здравствуйте! Меня зовут {person_name}, я {person_role} в компании {company_name}. Вы в поисках курса?
Сотрудник: Да, я бы хотел курс по SQL
{person_name}: Отлично, тогда вам идеально подойдёт курс {course}. Удачи в изучении!

Пример, рекомендации курс {course}:
{person_name}: Здравствуйте! Меня зовут {person_name}, я {person_role} в компании {company_name}. Что за курс вы хотите пройти?
Сотрудник: Я хочу научиться писать нейросети, но я ничего о них не знаю.
{person_name}: Хорошо. Какие навыки вы бы хотели получить после прохождения курса ?
Сотрудник: Научиться писать нейросети , машинное обучение на python, при этом я уже имею опыт писания программ на python.
{person_name}: Превосходно, как вам курс {course}, он подходит вам ?
Сотрудник: Да
{person_name}: Я рада, что смогла помочь вам с выбором курса, удачи в обучении.

Пример, рекомендации курс {course}, но он не соответствовал ожиданию сотрудника:
{person_name}: Здравствуйте! Меня зовут {person_name}, я {person_role} в компании {company_name}. Что за курс вы хотите пройти?
Сотрудник: Я хочу научиться писать нейросети, но я ничего о них не знаю.
{person_name}: Хорошо. Какие навыки вы бы хотели получить после прохождения курса ?
Сотрудник: Научиться писать нейросети , машинное обучение на python, при этом я уже имею опыт писания программ на python около двух лет.
{person_name}: Превосходно, как вам курс {course}, он подходит вам ?
Сотрудник: Нет, это не то.
{person_name}: Простите, вероятно вашим ожиданием больше соответствует курс "Машинное обучение – продвинутый курс". В нём есть темы про нейросети.
Сотрудник: Да.
{person_name}: Я рада, что смогла помочь вам с выбором курса, удачи в обучении.

Пример когда в подборке нет подходящего курса:
{person_name}: Здравствуйте! Меня зовут {person_name}, я {person_role} в компании {company_name}. Вы в поисках курса ?
Сотрудник: Да, я бы хотел какой-нибудь курс по 3D моделированию
{person_name}: К сожалению такого курса в нашей подборке нет.


Примеры того, что вам нельзя писать:
{person_name}: не можем вам этого дать
{person_name}: Вы ищите курс для себя?
{person_name}: Чтобы записаться на курс, просто нажмите на кнопку "Записаться" и заполните необходимую информацию.
{person_name}: Если у вас возникнут вопросы, пожалуйста, не стесняйтесь спрашивать.
{person_name}: Прошу прощения, если я перебила


""")]

    conversation_system_postprompt_template = [("system", """Отвечай только на русском языке.
Пиши только русскими буквами.
Перепроверяй себя на следования всем правилам, которые даны тебе выше.

Текущая стадия разговора:
{conversation_stage}

{person_name}:
""")]

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, '1')

    def seed_agent(self):
        self.current_conversation_stage = self.retrieve_conversation_stage('1')
        self.analyzer_history = copy.deepcopy(self.analyzer_history_template)
        self.analyzer_history.append(("user", "Привет"))
        self.conversation_history = copy.deepcopy(self.conversation_history_template)
        self.conversation_history.append(("user", "Привет"))

    def human_step(self, human_message):
        self.analyzer_history.append(("user", human_message))
        self.conversation_history.append(("user", human_message))
        self.human_messages += human_message

    def recommend_course_by_description(self, user_input, courses):
        courses = courses.values
        best_match = None
        best_match_score = 0
        for course in courses:
            description = course[1][2] + ''.join(course[1][3]) + ''.join(course[1][4])
            score = fuzz.ratio(user_input.lower(), description.lower())
            if score > best_match_score:
                best_match = course[0]
                best_match_score = score

        return best_match

    def ai_step(self):
        return self._call(inputs={})

    def analyse_stage(self):
        messages = self.analyzer_history + self.analyzer_system_postprompt_template
        template = ChatPromptTemplate.from_messages(messages)
        messages = template.format_messages()

        response = llm.invoke(messages)
        self.current_conversation_stage = (re.findall(r'\b\d+\b', response.content) + ['1'])[0]

        # self.current_conversation_stage = self.retrieve_conversation_stage(conversation_stage_id)
        print(f"[Этап разговора {self.current_conversation_stage}]")  #: {self.current_conversation_stage}")

    def _call(self, inputs: Dict[str, Any]) -> None:
        if self.current_conversation_stage == "3":
            self.course = str(self.recommend_course_by_description(''.join(self.human_messages), self.courses))
            print(self.course)
        messages = self.conversation_history + self.conversation_system_postprompt_template
        template = ChatPromptTemplate.from_messages(messages)
        messages = template.format_messages(
            person_name=self.person_name,
            person_role=self.person_role,
            company_name=self.company_name,
            course=self.course,
            courses_name=self.courses["Course_name"],
            conversation_purpose=self.conversation_purpose,
            conversation_stage=self.current_conversation_stage,
            conversation_type=self.conversation_type,
        )

        response = llm.invoke(messages)
        ai_message = (response.content).split('\n')[0]

        self.analyzer_history.append(("user", ai_message))
        self.conversation_history.append(("ai", ai_message))

        return ai_message

    @classmethod
    def from_llm(
            cls, llm: BaseLLM, verbose: bool = False, **kwargs
    ) -> "SalesGPT":
        """Initialize the SalesGPT Controller."""

        return cls(
            verbose=verbose,
            **kwargs,
        )
