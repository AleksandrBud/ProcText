from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, Dispatcher
from model_generate_congratelation import generate_text


class HappyNewYearBot:

    def __init__(self, token, model_dir, log_dir):

        def startCommand(update, context):
            context.bot.send_message(chat_id=update.message.chat_id,
                                     text='Привет! Я помогу тебе поздравить близких, ' 
                                     'родных и друзей c новым годом. Набери слово '
                                     'или фразу c которой начнется твое поздравление.')

        def textMessage(update, context):
            response = 'Ожидайте' # формируем текст ответа
            context.bot.send_message(chat_id=update.message.chat_id, text=response)
            response = self.generator .generate_text(update.message.text)
            with open(self.log_dir , 'a') as f:
              f.write(str(update.message.chat_id)
                        + '\n' + update.message.text
                        + '\n' + response
                        + '\n' + '\n')
            context.bot.send_message(chat_id=update.message.chat_id, text=response)


        self.log_dir = log_dir
        self.generator = generate_text()
        self.generator.download_model(load_path=model_dir)
        #Настройки
        self.updater = Updater(token=token) # Токен API к Telegram
        dispatcher: Dispatcher = self.updater.dispatcher
        # Хендлеры
        start_command_handler = CommandHandler('start', startCommand)
        text_message_handler = MessageHandler(Filters.text, textMessage)
        # Добавляем хендлеры в диспетчер
        dispatcher.add_handler(start_command_handler)
        dispatcher.add_handler(text_message_handler)
        # self.model = generate_text.load_model(model_path)


    def start_bot(self):
        # Начинаем поиск обновлений
        self.updater.start_polling(clean=True)
        # Останавливаем бота, если были нажаты Ctrl + C
        self.updater.idle()
