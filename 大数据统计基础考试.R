library(corrplot)
library(ggplot2)
library(showtext)
library(mvstats)
library("latex2exp")
library(scales)
library(glmnet)
library(MASS)
library(sparseLDA)
library(caret)
library(e1071)
library(randomForest)
library(nnet)
library(glmnet)
library(pROC)
library(AppliedPredictiveModeling)
showtext.auto(enable=T)
par(family="STSong")

setwd("~/Documents/作业/2016《大数据统计基础》考试题")
#第一题
loandata=read.csv("LoanStats3c.csv",header=T,sep=",",skip=1)
loanamnt=loandata$loan_amnt

summary(loanamnt)
loanamnt=na.omit(loanamnt)
sd(loanamnt)

#分组
breaks=cut(loanamnt,((1:36)-0.5)*1000)
zongti=data.frame(amnt=loanamnt,breaks=breaks)
n=nrow(zongti)
fenzu=data.frame(table(zongti$breaks))
fenzu$Freq2=fenzu$Freq/n+0.0000000001

#简单随机抽样并计算样本质量
fun1=function(m){
  index=sample(1:n,m)
  yangben=zongti[index,]
  fenzu2=data.frame(table(yangben$breaks))
  fenzu2$Freq2=fenzu2$Freq/m+0.0000000001
  I=sum((fenzu$Freq2-fenzu2$Freq2)*log(fenzu$Freq2/fenzu2$Freq2))
  Q=exp(-I)
  return(Q)
}

set.seed(436)
m=c(100,1000,5000,10000)
result=data.frame(m,Q=sapply(m,fun1))

plot(x=result$m,y=result$Q,type="b",
     xlab="样本容量",
     ylab="样本质量")
#当样本容量是5000的时候样本质量已经接近1了，因此，运用样本容量为5000的样本进行分析就可以得到相对精确的结果，同时可以大大提高计算效率。

#第二题
data("segmentationOriginal")
names(segmentationOriginal)
summary(segmentationOriginal$Class)
train=segmentationOriginal[segmentationOriginal$Case=="Train",-2]
test=segmentationOriginal[segmentationOriginal$Case=="Test",-2]

#封装法——我使用了逐步回归的方法筛选变量
mod1=glm(Class~.,data=train,family=binomial(link='logit'))
mod2=stepAIC(mod1)
summary(mod2)
#使用逐步回归我们用AIC准则找到了42个变量。
mod3=rfe(train[,-2],train$Class,sizes=c(10,20,30,50,60),rfeControl = rfeControl(functions=rfFuncs,method='cv'))
mod3$optVariables
#使用随机森林方法找到了20个变量，最佳的变量个数应该在20到30之间。

#过滤法——使用的包是caret，第一步删除近似等于常量的变量，第二步删除相关度过高的变量，第三步标准化处理，第四步使用了caret包中的sbf函数，使用随机森林法筛选和响应变量相关性较高的变量。
train2=train[,-2]
nearzero=nearZeroVar(train2)
train2=train2[,-nearzero]
cor=cor(train2)
highcor=findCorrelation(cor,0.9)
train2=train2[,-highcor]
process=preProcess(train2,na.remove=F)
train2=predict(process,train2)
data.filter=sbf(train2,train[,2],
                sbfControl=sbfControl(functions=rfSBF,
                                      verbose=F,
                                      method='cv'))
train2=train2[,data.filter$optVariables]
names(train2)
#用筛选变量的方法，我筛选出了57个变量。

#多种方法比较
#mod1逻辑回归mod2逐步回归mod3随机森林mod4 lasso mod5 ridge mod6 lda
mod4=cv.glmnet(as.matrix(train[,-2]),train$Class,family="binomial",alpha=1)
mod4
plot(mod4)
coef(mod4)

mod5=cv.glmnet(as.matrix(train[,-2]),train$Class,family="binomial",alpha=0)
plot(mod5)
coef(mod5)

x2=normalize(as.matrix(train[,-2]))
x2=x2$Xc
y2=class.ind(train$Class)
mod6=sda(x2,y2,lambda=0.01)
mod6

pred1=predict(mod1,newdata=test)
rocobj1=roc(test$Class,pred1)
pre1=ifelse(pred1<=-0.8,"PS","WS")
ta1=data.frame(table(pre1,test$Class))
pred2=predict(mod2,newdata=test)
rocobj2=roc(test$Class,pred2)
pre2=ifelse(pred2<=-0.6,"PS","WS")
ta2=data.frame(table(pre2,test$Class))
pred3=predict(mod3,newdata=test)
pred3x=as.matrix(pred3[,2])
rocobj3=roc(test$Class,pred3x)
pre3=ifelse(pred3x<=0.609,"WS","PS")
ta3=data.frame(table(pre3,test$Class))
pred4=predict(mod4,newx=as.matrix(test[,-2]))
rocobj4=roc(test$Class,pred4)
pre4=ifelse(pred4<=-0.7,"PS","WS")
ta4=data.frame(table(pre4,test$Class))
pred5=predict(mod5,newx=as.matrix(test[,-2]))
rocobj5=roc(test$Class,pred5)
pre5=ifelse(pred5<=-0.4,"PS","WS")
ta5=data.frame(table(pre5,test$Class))
x2pre=normalize(as.matrix(test[,-2]))
x2pre=x2pre$Xc
pred6=predict(mod6,x2pre)
pred6x=as.matrix(pred6$posterior[,1])
rocobj6=roc(test$Class,pred6x)
pre6=ifelse(pred6x<=0.7,"WS","PS")
ta6=data.frame(table(pre6,test$Class))

par(mfrow=c(2,3))
plot(rocobj1,print.thres=TRUE,print.auc=TRUE)
legend("bottomright",legend=c("regression"))
plot(rocobj2,print.thres=TRUE,print.auc=TRUE)
legend("bottomright",legend=c("stepwise"))
plot(rocobj3,print.thres=TRUE,print.auc=TRUE)
legend("bottomright",legend=c("randomForest"))
plot(rocobj4,print.thres=TRUE,print.auc=TRUE)
legend("bottomright",legend=c("lasso"))
plot(rocobj5,print.thres=TRUE,print.auc=TRUE)
legend("bottomright",legend=c("ridge"))
plot(rocobj6,print.thres=TRUE,print.auc=TRUE)
legend("bottomright",legend=c("sparseLDA"))
par(mfrow=c(1,1))

dat=data.frame(method=rep(c("regression","stepwise","randomForest","lasso","ridge","sparseLDA"),each=4),test=c(ta1$Var2,ta2$Var2,ta3$Var2,ta4$Var2,ta5$Var2,ta6$Var2),pre=c(ta1$pre1,ta2$pre2,ta3$pre3,ta4$pre4,ta5$pre5,ta6$pre6),freq=c(ta1$Freq,ta2$Freq,ta3$Freq,ta4$Freq,ta5$Freq,ta6$Freq))
dat$freq2=dat$freq/1010
dat$freq3=paste(round(dat$freq2*100,2),"%",sep="")
dat$tp=paste(dat$test,dat$pre,sep="")
dat$tp=as.factor(dat$tp)
levels(dat$tp)=c("(PS,PS)","(PS,WS)","(WS,PS)","(WS,WS)")
dat$method=factor(dat$method,levels=c("regression","stepwise","randomForest","lasso","ridge","sparseLDA"))
dat$freq4=dat$freq2/2
dat$freq4[dat$tp=="(PS,WS)"]=dat$freq2[dat$tp=="(PS,PS)"]+dat$freq4[dat$tp=="(PS,WS)"]
dat$freq4[dat$tp=="(WS,PS)"]=dat$freq2[dat$tp=="(PS,PS)"]+dat$freq2[dat$tp=="(PS,WS)"]+dat$freq4[dat$tp=="(WS,PS)"]
dat$freq4[dat$tp=="(WS,WS)"]=dat$freq2[dat$tp=="(PS,PS)"]+dat$freq2[dat$tp=="(PS,WS)"]+dat$freq2[dat$tp=="(WS,PS)"]+dat$freq4[dat$tp=="(WS,WS)"]

ph=ggplot(data=dat,aes(x=method,y=freq2,fill=tp))+
  geom_bar(stat="identity")+
  scale_fill_manual(values=c("red","pink","lightblue","blue"))+
  labs(x="方法",y="频率",fill="结果",title="六种方法预测效果比较")+
  geom_text(aes(x=method,y=freq4,label=freq3))+
  theme_bw()+
  theme(panel.grid=element_blank(),
        axis.text.x=element_text(size=10),
        axis.title=element_text(size=18),
        plot.title=element_text(size=20),
        legend.key.width=unit(0.8,"cm"),
        legend.key=element_rect(colour='white',fill='white',size=1))
print(ph)
ggsave("六种方法预测效果比较.png",width=7,height=5)

t1=system.time(glm(Class~.,data=train,family=binomial(link='logit')))
t2=system.time(stepAIC(mod1))
t3=system.time(rfe(train[,-2],train$Class,sizes=c(10,20,30,50,60),rfeControl=rfeControl(functions=rfFuncs,method='cv')))
t4=system.time(glmnet(as.matrix(train[,-2]),train$Class,family="binomial",alpha=1,lambda=0.007))
t5=system.time(glmnet(as.matrix(train[,-2]),train$Class,family="binomial",alpha=0,lambda=0.02))
t6=system.time(sda(x2,y2,lambda=0.01))

#第三题
shouji=read.csv("shouji.csv",header=T,sep=",")
shouji=na.omit(shouji)
brand=shouji$brand
shouji=shouji[,-1]
#前22个问题为自变量，23到25的问题表示手机用户的满意度，26到28的问题表示手机用户的忠诚度
m=cor(shouji[,1:22]) 
corrplot(m,method="square",type="full",tl.col="black")
#由协方差矩阵图可知，1到22的问题明显的分成了三块，说明可能存在三个潜在因素，在后文中，我将使用因子分析的方法提出这三种因素。
fac.out=factpc(shouji[,1:22],3,rotation="varimax") 
fac.out$Vars
write.csv(round(fac.out$Vars,4),file="因子分析方差贡献率.csv",row.names=F)
#提取出的三个公因子的方差贡献率分别是26.00% 22.93%和8.46%，累计方差贡献率达到57.39%
data.frame(cbind(round(fac.out$loadings,2),round(fac.out$common,2)),row.names=colnames(shouji[,1:22]))
write.csv(round(data.frame(cbind(round(fac.out$loadings,2),round(fac.out$common,2)),row.names=colnames(shouji[,1:22])),4),file="因子载荷矩阵.csv",row.names=F)
#因子载荷矩阵显示，第一公因子主要和q11,q12,q13,q14,q15,q16,q17,q18,q19,q20相关，第二公因子主要和q1,q2,q3,q4,q5,q6,q7,q9,q10相关。第三公因子主要和q8相关。第一公因子对应象征价值，第二公因子对应用户体验，第三公因子对应性价比。

fac.out2=factpc(shouji[23:28],2,rotation="varimax") 
fac.out2$Vars
write.csv(round(fac.out2$Vars,4),file="满意度和忠诚度方差贡献率.csv",row.names=F)
#对q23到q28提取公因子，提取出的两个公因子方差贡献率分别为52.24% 23.41%方差累计贡献率是75.65%
data.frame(cbind(round(fac.out2$loadings,2),round(fac.out2$common,2)),row.names=colnames(shouji[23:28]))
write.csv(round(data.frame(cbind(round(fac.out2$loadings,2),round(fac.out2$common,2)),row.names=colnames(shouji[23:28])),4),file="满意度和忠诚度因子载荷矩阵.csv",row.names=F)
#第一公因子主要和q23 q24 q25相关 第二公因子主要和q26 q27 q28相关。第一公因子表示满意度，第二公因子表示忠诚度。

#回归分析评价用户体验、性价比和象征价值和满意度以及忠诚度之间的关系：
x=data.frame(fac.out$scores)
names(x)=c("象征价值","用户体验","性价比")
y=data.frame(fac.out2$scores)
names(y)=c("满意度","忠诚度")
dat=data.frame(x,y)
mod1=lm(data=dat,满意度~象征价值+用户体验+性价比)
mod2=lm(data=dat,忠诚度~象征价值+用户体验+性价比)
summary(mod1)
summary(mod2)

#从数据中可以发现不同品牌手机的用户体验之间的差别。
dat$品牌=brand
dat$品牌=as.factor(dat$品牌)
levels(dat$品牌)=c("三星","苹果","HTC","华为")
dat2=data.frame(x=dat$品牌,y=c(dat$象征价值,dat$用户体验,dat$性价比,dat$满意度,dat$忠诚度),z=rep(names(dat)[-1],each=nrow(dat)))
dat2$z=factor(dat2$z,levels=names(dat)[-1])
ph1=ggplot(data=dat2,aes(x=x,y=y))+
  geom_boxplot()+
  stat_summary(fun.y="mean",geom="point",
               shape=23,size=3,fill="white")+
  theme_bw()+
  labs(x="",y="")+
  scale_y_continuous(limits=c(-3,3))+
  theme(panel.grid=element_blank(),
        axis.text=element_text(size=15),
        axis.title=element_text(size=18))+
  facet_wrap(~z)
print(ph1)
ggsave("四种手机的比较.png",width=7,height=5)

#第四题
loandata=read.csv("LoanStats3c.csv",header=T,sep=",",skip=1)
summary(loandata)

#风玫瑰图设计思路：贷款数额，认证状态和信用评级。目的：观察在不同贷款数额下大家的认证状态和信用评级状态。
dat=data.frame(x1=loandata$loan_amnt,x2=loandata$verification_status,x3=loandata$grade)
dat=na.omit(dat)
dat$x4=cut(dat$x1,c(999,quantile(dat$x1)[-1]))
dat$x5=as.factor(dat$x4)
levels(dat$x5)=c("贷款金额低","贷款金额较低","贷款金额较高","贷款金额高")
dat$x6=as.factor(as.character(dat$x2))
levels(dat$x6)=c("未认证","收入认证","来源认证")
ph1=ggplot(data=dat,aes(x=x6,fill=x3))+
  geom_bar()+
  labs(x="认证状态",y="频数",fill="信用评级",title="认证状态和信用评级风玫瑰图")+
  coord_polar(theta="x",direction=-1)+
  facet_wrap(~x5)+
  theme_bw()+
  theme(axis.text.x=element_text(size=7),
        axis.title=element_text(size=15),
        plot.title=element_text(size=18),
        legend.key.width=unit(0.8,"cm"),
        legend.key=element_rect(colour='white',fill='white',size=1))
print(ph1)
ggsave("风玫瑰图.png",width=7,height=5)

#贷款金额分布直方图
x=loandata$loan_amnt
x=na.omit(x)
tmp=seq(from=999,to=35000,length=12)
dat=data.frame(x,breaks=cut(x,tmp))
dat2=data.frame(table(dat$breaks))
dat2$Freq2=dat2$Freq/sum(dat2$Freq)
dat2$Var2=(tmp[-1]+tmp[-12])/2

ph3=ggplot()+
  annotate("rect",xmin=500,xmax=35500,ymin=-0.005,ymax=0.2,color="black",fill=muted("gray"),size=2)+
  geom_bar(data=dat2,aes(x=Var2,y=Freq2),
           stat="identity",fill="white")+
  geom_smooth(data=dat2,aes(x=Var2,y=Freq2),
              color="black",se=F,size=0.6,span=0.5)+
  geom_text(data=dat2,aes(x=Var2,y=Freq2,label=Freq),vjust=1.5)+
  labs(x="贷款金额",y="密度",title="贷款金额直方图")+
  theme_classic()+
  annotate("text",x=11000,y=0.19,label="此处密度最大",size=6,color="white")+
  theme(axis.text.x=element_text(size=15),
        axis.text.y=element_text(size=15),
        axis.title=element_text(size=18),
        plot.title=element_text(size=20),
        axis.ticks=element_blank())
print(ph3)
ggsave("贷款金额分布直方图.png",width=7,height=5)

#贷款金额和年收入密度图
dat=data.frame(x1=loandata$loan_amnt,x2=loandata$annual_inc)
dat=na.omit(dat)
summary(dat)
dat2=data.frame(x=c(dat$x1,dat$x2),y=rep(c("贷款金额","年收入"),each=nrow(dat)))

ph4=ggplot(data=dat2,aes(x=x,fill=y))+
  geom_density(alpha=0.5,adjust=1.5)+
  annotate("rect",xmin=1000,xmax=25000,ymin=4e-5,ymax=6.1e-5,fill="gray",alpha=0.5)+
  annotate("text",x=40000,y=5.5e-5,label="此处贷款金额密度最大",color="black",size=6)+
  scale_x_continuous(limits=c(1000,100000))+
  scale_y_continuous(breaks=c(0,2e-5,4e-5,6e-5),
                     labels=c(TeX("$ \\0.0 \\times 10^{-5}$"),
                              TeX("$ \\2.0 \\times 10^{-5}$"),
                              TeX("$ \\4.0 \\times 10^{-5}$"),
                              TeX("$ \\6.0 \\times 10^{-5}$")))+
  labs(x="美元",y="密度",fill="",title="贷款金额和年收入密度曲线")+
  theme_bw()+
  theme(axis.text=element_text(size=15),
        axis.title=element_text(size=18),
        plot.title=element_text(size=20),
        panel.grid=element_blank(),
        legend.position=c(0.7,0.7),
        legend.key.width=unit(1,"cm"),
        legend.key=element_rect(colour='white',fill='white',size=1),
        legend.text=element_text(size=15),
        legend.key.size=unit(0.7,'cm'))
print(ph4)
ggsave("贷款金额和年收入密度曲线.png",width=7,height=5)

#热图
set.seed(436)
index=sample(1:nrow(loandata),100)
dat=data.frame(x1=loandata$loan_amnt,x2=loandata$installment,x3=loandata$annual_inc,x4=loandata$dti)
dat=dat[index,]
dat=na.omit(dat)
summary(dat)
dat2=scale(dat)
dis=dist(dat2,method="euclidean")
dis=as.matrix(dis)
heatmap(dis,main="聚类分析热图")