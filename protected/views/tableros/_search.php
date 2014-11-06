<div class="wide form">

<?php $form=$this->beginWidget('CActiveForm', array(
        'action'=>Yii::app()->createUrl($this->route),
        'method'=>'get',
)); ?>

       
<div class="row">
                <?php echo $form->label($model,'Tablero'); ?>
                <?php echo $form->textField($model,'Tablero',array('size'=>50,'maxlength'=>50)); ?>
        </div>
        <div class="row">
            
             <div class="row">
                <?php echo $form->label($model,'TAG'); ?>
                <?php echo $form->textField($model,'TAG',array('size'=>50,'maxlength'=>50)); ?>
        </div>
                <?php echo $form->label($model,'Proceso'); ?>
                <?php echo $form->textField($model,'Proceso',array('size'=>50,'maxlength'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Area'); ?>
                <?php echo $form->textField($model,'Area',array('size'=>50,'maxlength'=>50)); ?>
        </div>

       

        

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Buscar')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
