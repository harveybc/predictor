<div class="wide form">

<?php $form=$this->beginWidget('CActiveForm', array(
        'action'=>Yii::app()->createUrl($this->route),
        'method'=>'get',
)); ?>

        <div class="row">
                <?php echo $form->label($model,'id'); ?>
                <?php echo $form->textField($model,'id',array('size'=>20,'maxlength'=>20)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Campo0'); ?>
                <?php echo $form->textArea($model,'Campo0',array('rows'=>6, 'cols'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Campo1'); ?>
                <?php echo $form->textArea($model,'Campo1',array('rows'=>6, 'cols'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Campo2'); ?>
                <?php echo $form->textArea($model,'Campo2',array('rows'=>6, 'cols'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Campo3'); ?>
                <?php echo $form->textArea($model,'Campo3',array('rows'=>6, 'cols'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Campo4'); ?>
                <?php echo $form->textArea($model,'Campo4',array('rows'=>6, 'cols'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Campo5'); ?>
                <?php echo $form->textArea($model,'Campo5',array('rows'=>6, 'cols'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Campo6'); ?>
                <?php echo $form->textArea($model,'Campo6',array('rows'=>6, 'cols'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Campo7'); ?>
                <?php echo $form->textArea($model,'Campo7',array('rows'=>6, 'cols'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Campo8'); ?>
                <?php echo $form->textArea($model,'Campo8',array('rows'=>6, 'cols'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Campo9'); ?>
                <?php echo $form->textArea($model,'Campo9',array('rows'=>6, 'cols'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Campo10'); ?>
                <?php echo $form->textArea($model,'Campo10',array('rows'=>6, 'cols'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Campo11'); ?>
                <?php echo $form->textArea($model,'Campo11',array('rows'=>6, 'cols'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Campo12'); ?>
                <?php echo $form->textArea($model,'Campo12',array('rows'=>6, 'cols'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Campo13'); ?>
                <?php echo $form->textArea($model,'Campo13',array('rows'=>6, 'cols'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Campo14'); ?>
                <?php echo $form->textArea($model,'Campo14',array('rows'=>6, 'cols'=>50)); ?>
        </div>

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Search')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
