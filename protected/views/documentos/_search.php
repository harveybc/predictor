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
                <?php echo $form->label($model,'permitirAdiciones'); ?>
                <?php echo $form->checkBox($model,'permitirAdiciones'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'permitirAnotaciones'); ?>
                <?php echo $form->checkBox($model,'permitirAnotaciones'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'autorizarOtros'); ?>
                <?php echo $form->checkBox($model,'autorizarOtros'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'requiereAutorizacion'); ?>
                <?php echo $form->checkBox($model,'requiereAutorizacion'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'secuencia'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'ordenSecuencia'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'eliminado'); ?>
                <?php echo $form->checkBox($model,'eliminado'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'conservacionInicio'); ?>
                <?php $this->widget('zii.widgets.jui.CJuiDatePicker',
						 array(
								 'model'=>'$model',
								 'name'=>'Documentos[conservacionInicio]',
								 //'language'=>'de',
								 'value'=>$model->conservacionInicio,
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
                <?php echo $form->label($model,'conservacionFin'); ?>
                <?php $this->widget('zii.widgets.jui.CJuiDatePicker',
						 array(
								 'model'=>'$model',
								 'name'=>'Documentos[conservacionFin]',
								 //'language'=>'de',
								 'value'=>$model->conservacionFin,
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
                <?php echo $form->label($model,'conservacionPermanente'); ?>
                <?php echo $form->checkBox($model,'conservacionPermanente'); ?>
        </div>

        

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Buscar')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
