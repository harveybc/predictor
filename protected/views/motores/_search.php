<div class="wide form">

<?php $form=$this->beginWidget('CActiveForm', array(
        'action'=>Yii::app()->createUrl($this->route),
        'method'=>'get',
)); ?>

        <div class="row">
                <?php echo $form->label($model,'Codigo'); ?>
                <?php echo $form->textField($model,'Codigo',array('size'=>50,'maxlength'=>50)); ?>
        </div>       

        <div class="row">
                <?php echo $form->label($model,'TAG'); ?>
                <?php echo $form->textField($model,'TAG',array('size'=>50,'maxlength'=>50)); ?>
        </div>

       
        <div class="row">
                <?php echo $form->label($model,'Marca'); ?>
                <?php echo $form->textField($model,'Marca',array('size'=>50,'maxlength'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Modelo'); ?>
                <?php echo $form->textField($model,'Modelo',array('size'=>50,'maxlength'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Serie'); ?>
                <?php echo $form->textField($model,'Serie',array('size'=>50,'maxlength'=>50)); ?>
        </div>

       
        <div class="row">
                <?php echo $form->label($model,'Lubricante'); ?>
                <?php echo $form->textField($model,'Lubricante',array('size'=>50,'maxlength'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Motor'); ?>
                <?php echo $form->textField($model,'Motor',array('size'=>60,'maxlength'=>255)); ?>
        </div>
        
         <div class="row">
                <?php echo $form->label($model,'Equipo'); ?>
                <?php echo $form->textField($model,'Equipo',array('size'=>60,'maxlength'=>255)); ?>
        </div>

       

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Buscar')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
