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
                <?php echo $form->label($model,'descripcion'); ?>
                <?php echo $form->textField($model,'descripcion',array('size'=>60,'maxlength'=>128)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'fechaInicio'); ?>
                <?php $this->widget('zii.widgets.jui.CJuiDatePicker',
						 array(
								 'model'=>'$model',
								 'name'=>'EvaluacionesGenerales[fechaInicio]',
								 //'language'=>'de',
								 'value'=>$model->fechaInicio,
								 'htmlOptions'=>array('size'=>10, 'style'=>'width:80px !important'),
									 'options'=>array(
									 'showButtonPanel'=>true,
									 'changeYear'=>true,                                      
									 'changeYear'=>true,
									 ),
								 )
							 );
					; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'fechaFin'); ?>
                <?php $this->widget('zii.widgets.jui.CJuiDatePicker',
						 array(
								 'model'=>'$model',
								 'name'=>'EvaluacionesGenerales[fechaFin]',
								 //'language'=>'de',
								 'value'=>$model->fechaFin,
								 'htmlOptions'=>array('size'=>10, 'style'=>'width:80px !important'),
									 'options'=>array(
									 'showButtonPanel'=>true,
									 'changeYear'=>true,                                      
									 'changeYear'=>true,
									 ),
								 )
							 );
					; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'pregunta1'); ?>
                <?php echo $form->textField($model,'pregunta1'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'pregunta2'); ?>
                <?php echo $form->textField($model,'pregunta2'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'pregunta3'); ?>
                <?php echo $form->textField($model,'pregunta3'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'pregunta4'); ?>
                <?php echo $form->textField($model,'pregunta4'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'pregunta5'); ?>
                <?php echo $form->textField($model,'pregunta5'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'pregunta6'); ?>
                <?php echo $form->textField($model,'pregunta6'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'pregunta7'); ?>
                <?php echo $form->textField($model,'pregunta7'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'pregunta8'); ?>
                <?php echo $form->textField($model,'pregunta8'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'pregunta9'); ?>
                <?php echo $form->textField($model,'pregunta9'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'pregunta10'); ?>
                <?php echo $form->textField($model,'pregunta10'); ?>
        </div>

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Buscar')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
