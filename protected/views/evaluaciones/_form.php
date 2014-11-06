<style type="text/css">



    .secuencias{

        width: 200px;
        
        
    }



</style>

<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>
<div styler="width:630px!important;margin-left:0px;margin-bottom:5px !important;border-color:#961C1F;padding-top:10px;padding-right:5px;padding-left:14px;">

	<div>
            <div>
                <div>
                    <div class="row">
		<?php echo $form->labelEx($model,'fecha'); ?>
<?php $this->widget('zii.widgets.jui.CJuiDatePicker',
						 array(
								 'model'=>'$model',
								 'name'=>'Evaluaciones[fecha]',
								 //'language'=>'de',
								 'value'=>$model->fecha,
								 'htmlOptions'=>array('size'=>10, 'style'=>'width:100px !important'),
									 'options'=>array(
									 'showButtonPanel'=>true,
									 'changeYear'=>true,                                      
									 'changeYear'=>true,
									 ),
								 )
							 );
					; ?>
<?php echo $form->error($model,'fecha'); ?>
	</div>

                    </div>
                     <div>
                         <label for="Usuarios">Evaluación realizada por el usuario</label><?php 
					$this->widget('application.components.Relation', array(
							'model' => $model,
							'relation' => 'usuario0',
							'fields' => 'Username',
							'allowEmpty' => false,
							'style' => 'dropdownlist',
                                                        'htmlOptions' => array(
                                                              'class'=>'secuencias',)
							)
						); ?>
			
                    </div>
                     <div>
                         <label for="EvaluacionesGenerales">Evaluación General realizada por el usuario </label><?php 
					$this->widget('application.components.Relation', array(
							'model' => $model,
							'relation' => 'evaluacionGeneral0',
							'fields' => 'descripcion',
							'allowEmpty' => true,
							'style' => 'dropdownlist',
                                                        'htmlOptions' => array(
                                                                'class'=>'secuencias',)
							)
						); ?>
                    </div>
                </div>
            </div>
	
</div>	

<div>
<div >
    <div>
        <div class="row">
		<?php echo $form->labelEx($model,'pregunta1'); ?>
<?php echo $form->textField($model,'pregunta1',array('style' => 'width:600px')); ?>
<?php echo $form->error($model,'pregunta1'); ?>
	</div>

	
        </div>
        </div>
        <div >
    <div>
        <div class="row">
		<?php echo $form->labelEx($model,'pregunta2',array('style' => 'width:600px')); ?>
<?php echo $form->textField($model,'pregunta2'); ?>
<?php echo $form->error($model,'pregunta2'); ?>
	</div>
        </div>
        </div>
        <div >
    <div>
        <div class="row">
		<?php echo $form->labelEx($model,'pregunta3',array('style' => 'width:600px')); ?>
<?php echo $form->textField($model,'pregunta3'); ?>
<?php echo $form->error($model,'pregunta3'); ?>
	</div>

	
        </div>
        </div>
        <div >
    <div>
        <div class="row">
		<?php echo $form->labelEx($model,'pregunta4',array('style' => 'width:600px')); ?>
<?php echo $form->textField($model,'pregunta4'); ?>
<?php echo $form->error($model,'pregunta4'); ?>
	</div>

	
        </div>
        </div>
        <div >
    <div>
        <div class="row">
		<?php echo $form->labelEx($model,'pregunta5',array('style' => 'width:600px')); ?>
<?php echo $form->textField($model,'pregunta5'); ?>
<?php echo $form->error($model,'pregunta5'); ?>
	</div>
        </div>
        </div>
        <div >
    <div>
        <div class="row">
		<?php echo $form->labelEx($model,'pregunta6',array('style' => 'width:600px')); ?>
<?php echo $form->textField($model,'pregunta6'); ?>
<?php echo $form->error($model,'pregunta6'); ?>
	</div>

	
        </div>
        </div>
        <div >
    <div>
        <div class="row">
		<?php echo $form->labelEx($model,'pregunta7',array('style' => 'width:600px')); ?>
<?php echo $form->textField($model,'pregunta7'); ?>
<?php echo $form->error($model,'pregunta7'); ?>
	</div>

	
        </div>
        </div>
        <div >
    <div>
        <div class="row">
		<?php echo $form->labelEx($model,'pregunta8',array('style' => 'width:600px')); ?>
<?php echo $form->textField($model,'pregunta8'); ?>
<?php echo $form->error($model,'pregunta8'); ?>
	</div>

	
        </div>
        </div>
        <div >
    <div>
        <div class="row">
		<?php echo $form->labelEx($model,'pregunta9',array('style' => 'width:600px')); ?>
<?php echo $form->textField($model,'pregunta9'); ?>
<?php echo $form->error($model,'pregunta9'); ?>
	</div>

	
        </div>
        </div>
        <div >
    <div>
        <div class="row">
		<?php echo $form->labelEx($model,'pregunta10',array('style' => 'width:600px')); ?>
<?php echo $form->textField($model,'pregunta10'); ?>
<?php echo $form->error($model,'pregunta10'); ?>
	</div>
        </div>
        </div>
</div>

			